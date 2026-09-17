// Compiles a ResourcePlan's reachable SRT expression graph into a flat, topologically-ordered
// CompiledSrtProgram once per unique shader, so per-draw evaluation (ExecuteSrtProgram) is a
// single forward loop over an array instead of SrtWalker.cpp's Evaluator recursively re-walking
// the same Inst graph, node by node, through a switch on every single draw.
//
// This file is a deliberately case-by-case port of Evaluator::EvaluateInst/EvaluateRawRead/
// EvaluateExtract (SrtWalker.cpp) split into two halves per opcode: a Compile-time half (this
// file's Compiler class) that resolves everything structural/static once -- operand slots,
// MemoryFlags, SRT slot indices, component indices, the "is this SRT slot clean" flag -- and an
// execute-time half (ExecuteSrtProgram) that does only the genuinely runtime-dependent work
// (user_data lookups, guest memory reads, and the arithmetic itself). Every case here should be
// read side-by-side with its EvaluateInst counterpart; anything that doesn't map cleanly compiles
// to AlwaysFails rather than guessing, matching EvaluateInst's own `default: break;`.
//
// Only opcodes SrtWalker.cpp's own Evaluator actually resolves are handled here (e.g. no
// CompositeExtractU32x4/BitCount32/FindILsb32/FindUMsb32 -- this codebase's Evaluator has no
// cases for them either, so they already compile to AlwaysFails via the default path, matching
// the interpreter exactly). SelectU32/U1/F32 gets its own CompiledOpKind rather than folding into
// the generic arithmetic dispatch: this codebase's Select resolves its predicate through the
// clean pass (when one is available) and only evaluates the branch it actually selects, unlike a
// plain eager ternary -- see CompiledOpKind::Select's case in ExecutePass below.
//
// Two facts from ExtractResourcePlan (ResourceMaterialization.cpp) shape this compiler:
//   - Its Clone lambda already resolves every Phi to its invariant value or leaves it unreachable,
//     so any Phi that still appears in a ResourcePlan's value_storage is provably non-invariant.
//     No runtime Phi-resolution logic is needed here at all -- see CompiledOpKind::AlwaysFails.
//   - CompositeConstructU32x2 and IAddCarry32 are never given their own EvaluateInst case -- they
//     are only ever reached through CompositeExtractU32x2's special-casing, which reads directly
//     through them without evaluating the construct/carry node as its own value. This compiler
//     mirrors that with ExtractPassthrough/ExtractCarryHalf, so a construct or carry node reached
//     any other way still compiles to AlwaysFails, exactly like today's `default: break;`.
#include "graphics/shader/recompiler/ir/passes/SrtCompiler.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

// Sentinel stored in m_slot_for_inst while an Inst's operands are still being compiled, so a
// genuine cycle (should be impossible post-Phi-elimination, but never trust that blindly) fails
// closed as AlwaysFails instead of infinitely recursing.
constexpr uint32_t kInProgressSlot = UINT32_MAX - 1;

bool IsRawReadOpcode(const ResourcePlan& program, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= program.memory_info.size()) {
		return false;
	}
	const auto kind = program.memory_info[index].kind;
	return (op == ValueOpcode::LoadAddressU32 && kind == ResourceKind::ScalarAddress) ||
	       (op == ValueOpcode::ReadConstBuffer && kind == ResourceKind::ScalarBuffer);
}

class Compiler {
public:
	explicit Compiler(const ResourcePlan& program): m_program(program) {}

	CompiledSrtProgram Run() {
		CompiledSrtProgram result;
		result.descriptor_source_slots.resize(m_program.descriptor_sources.size());
		for (size_t i = 0; i < m_program.descriptor_sources.size(); i++) {
			const auto& source = m_program.descriptor_sources[i];
			for (uint32_t d = 0; d < source.dword_count; d++) {
				result.descriptor_source_slots[i][d] = CompileValue(source.dwords[d]);
			}
		}
		result.srt_read_slots.resize(m_program.srt_reads.size());
		for (size_t i = 0; i < m_program.srt_reads.size(); i++) {
			result.srt_read_slots[i] = CompileValue(m_program.srt_reads[i].value);
		}
		result.control_flow_condition_slots.resize(m_program.control_flow.size());
		for (size_t i = 0; i < m_program.control_flow.size(); i++) {
			const auto& condition = m_program.control_flow[i].condition;
			result.control_flow_condition_slots[i] =
			    condition.IsEmpty() ? kInvalidSlot : CompileValue(condition);
		}
		result.ops = std::move(m_ops);
		result.needs_clean_pass =
		    !m_program.control_flow.empty() ||
		    std::ranges::any_of(m_program.clean_flat_slots, [](uint8_t clean) { return clean != 0u; });
		return result;
	}

private:
	static CompiledOp AlwaysFailsOp() {
		CompiledOp op;
		op.kind = CompiledOpKind::AlwaysFails;
		return op;
	}

	// Post-order: every operand this node references is compiled (and therefore already has a
	// lower slot index in m_ops) before this node's own CompiledOp is appended -- the invariant
	// ExecuteSrtProgram's single forward pass depends on.
	uint32_t CompileValue(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return EmitConstant(value);
		}
		auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return EmitOp(AlwaysFailsOp());
		}
		if (const auto it = m_slot_for_inst.find(inst); it != m_slot_for_inst.end()) {
			if (it->second == kInProgressSlot) {
				return EmitOp(AlwaysFailsOp());
			}
			return it->second;
		}
		m_slot_for_inst.emplace(inst, kInProgressSlot);
		auto       op          = CompileInst(*inst);
		const auto slot        = EmitOp(std::move(op));
		m_slot_for_inst[inst] = slot;
		return slot;
	}

	uint32_t EmitOp(CompiledOp op) {
		m_ops.push_back(std::move(op));
		return static_cast<uint32_t>(m_ops.size() - 1);
	}

	uint32_t EmitConstant(Value value) {
		CompiledOp op;
		op.kind = CompiledOpKind::Constant;
		switch (value.GetType()) {
			case Type::U1: op.immediate = value.U1(); break;
			case Type::U8: op.immediate = value.U8(); break;
			case Type::U16: op.immediate = value.U16(); break;
			case Type::U32: op.immediate = value.U32(); break;
			case Type::U64: op.immediate = value.U64(); break;
			case Type::F32: op.immediate = std::bit_cast<uint32_t>(value.F32Value()); break;
			default: return EmitOp(AlwaysFailsOp());
		}
		return EmitOp(std::move(op));
	}

	// Mirrors Evaluator::EvaluateInst case-by-case -- see this file's header comment.
	CompiledOp CompileInst(const Inst& inst) {
		const auto opcode = inst.GetOpcode();
		switch (opcode) {
			case ValueOpcode::GetUserData: {
				if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != Type::ScalarReg) {
					return AlwaysFailsOp();
				}
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind      = CompiledOpKind::GetUserData;
				op.immediate = reg - m_program.user_data_base;
				return op;
			}
			case ValueOpcode::GetShaderBase: {
				CompiledOp op;
				op.kind = CompiledOpKind::GetShaderBase;
				return op;
			}
			case ValueOpcode::Phi:
				// Provably non-invariant -- see this file's header comment.
				return AlwaysFailsOp();
			case ValueOpcode::ReadFirstLane: {
				if (inst.NumArgs() != 2) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind         = CompiledOpKind::ReadFirstLane;
				op.num_operands = 2;
				op.operands[0]  = CompileValue(inst.Arg(0));
				op.operands[1]  = CompileValue(inst.Arg(1));
				return op;
			}
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
				return GenericUnary(inst, opcode);
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2:
				return CompileExtract(inst, opcode);
			case ValueOpcode::CompositeConstructU64:
				return GenericBinary(inst, opcode);
			case ValueOpcode::ReadConst: {
				if (inst.NumArgs() != 2) {
					return AlwaysFailsOp();
				}
				const auto slot = inst.Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != Type::U32) {
					return AlwaysFailsOp();
				}
				const auto slot_index = slot.U32();
				if (slot_index >= m_program.srt_reads.size()) {
					return AlwaysFailsOp();
				}
				CompiledOp op;
				op.kind           = CompiledOpKind::ReadConst;
				op.srt_slot_clean = slot_index < m_program.clean_flat_slots.size() &&
				                    m_program.clean_flat_slots[slot_index] != 0u;
				op.num_operands = 1;
				// The SRT read's own value graph -- compiled once here (shared with anyone else
				// referencing the same srt_reads[slot_index], since CompileValue dedupes by Inst*
				// either way).
				op.operands[0] = CompileValue(m_program.srt_reads[slot_index].value);
				return op;
			}
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
				if (IsRawReadOpcode(m_program, inst)) {
					return CompileRawRead(inst, opcode);
				}
				return AlwaysFailsOp();
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMin32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor:
				return GenericBinary(inst, opcode);
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::LogicalNot:
				return GenericUnary(inst, opcode);
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
				return GenericTernary(inst, opcode);
			case ValueOpcode::BitFieldInsert:
				return GenericQuaternary(inst, opcode);
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectF32:
				return CompileSelect(inst);
			case ValueOpcode::UndefU1:
			case ValueOpcode::UndefU8:
			case ValueOpcode::UndefU16:
			case ValueOpcode::UndefU32:
			case ValueOpcode::UndefU64:
			default:
				return AlwaysFailsOp();
		}
	}

	CompiledOp GenericUnary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 1) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 1;
		op.operands[0]  = CompileValue(inst.Arg(0));
		return op;
	}

	CompiledOp GenericBinary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 2) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 2;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		return op;
	}

	CompiledOp GenericTernary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 3) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 3;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		op.operands[2]  = CompileValue(inst.Arg(2));
		return op;
	}

	CompiledOp GenericQuaternary(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() < 4) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Generic;
		op.opcode       = opcode;
		op.num_operands = 4;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		op.operands[2]  = CompileValue(inst.Arg(2));
		op.operands[3]  = CompileValue(inst.Arg(3));
		return op;
	}

	// Mirrors EvaluateInst's Select case: the predicate is resolved through the clean pass (see
	// ExecutePass), and only the branch it selects is ever compiled to require a value -- but
	// both branches still need their own slots compiled here so the flat array stays complete for
	// anyone else referencing the same nodes.
	CompiledOp CompileSelect(const Inst& inst) {
		if (inst.NumArgs() != 3) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind         = CompiledOpKind::Select;
		op.num_operands = 3;
		op.operands[0]  = CompileValue(inst.Arg(0));
		op.operands[1]  = CompileValue(inst.Arg(1));
		op.operands[2]  = CompileValue(inst.Arg(2));
		return op;
	}

	// Mirrors Evaluator::EvaluateExtract.
	CompiledOp CompileExtract(const Inst& inst, ValueOpcode opcode) {
		if (inst.NumArgs() != 2) {
			return AlwaysFailsOp();
		}
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32) {
			return AlwaysFailsOp();
		}
		const auto component = index.U32();
		if (component >= 2u) {
			return AlwaysFailsOp();
		}
		if (opcode == ValueOpcode::CompositeExtractU64) {
			CompiledOp op;
			op.kind         = CompiledOpKind::ExtractU64;
			op.component    = component;
			op.num_operands = 1;
			op.operands[0]  = CompileValue(inst.Arg(0));
			return op;
		}
		const auto* source = inst.Arg(0).ResolveInstruction();
		if (source == nullptr) {
			return AlwaysFailsOp();
		}
		const auto source_opcode = source->GetOpcode();
		if (source_opcode == ValueOpcode::CompositeConstructU32x2) {
			// EvaluateExtract never evaluates the construct node itself -- it reaches straight
			// through to one of its arguments. Guard against a construct with too few args
			// (source->Arg(component) would otherwise assert inside Inst::Arg).
			if (component >= source->NumArgs()) {
				return AlwaysFailsOp();
			}
			CompiledOp op;
			op.kind         = CompiledOpKind::ExtractPassthrough;
			op.num_operands = 1;
			op.operands[0]  = CompileValue(source->Arg(component));
			return op;
		}
		if (source_opcode == ValueOpcode::IAddCarry32) {
			if (source->NumArgs() < 2) {
				return AlwaysFailsOp();
			}
			CompiledOp op;
			op.kind         = CompiledOpKind::ExtractCarryHalf;
			op.component    = component;
			op.num_operands = 2;
			op.operands[0]  = CompileValue(source->Arg(0));
			op.operands[1]  = CompileValue(source->Arg(1));
			return op;
		}
		return AlwaysFailsOp();
	}

	// Mirrors Evaluator::EvaluateRawRead's *structural* validation -- everything here is static
	// (memory_info, handle shape, immediate offset), so it's fully resolved once, at compile
	// time. Only the address arithmetic and the actual guest read remain for execute time.
	CompiledOp CompileRawRead(const Inst& inst, ValueOpcode opcode) {
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			return AlwaysFailsOp();
		}
		const auto& mem = m_program.memory_info[flags.index];
		if (inst.NumArgs() < 2) {
			return AlwaysFailsOp();
		}
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr) {
			return AlwaysFailsOp();
		}
		const bool is_const_buffer_read = opcode == ValueOpcode::ReadConstBuffer;
		if (is_const_buffer_read && handle->NumArgs() != 4u) {
			return AlwaysFailsOp();
		}
		if (handle->NumArgs() < 2) {
			return AlwaysFailsOp();
		}
		CompiledOp op;
		op.kind                 = CompiledOpKind::RawRead;
		op.is_const_buffer_read = is_const_buffer_read;
		op.immediate = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(mem.offset)));
		op.num_operands = is_const_buffer_read ? 5 : 3;
		op.operands[0]  = CompileValue(handle->Arg(0)); // low
		op.operands[1]  = CompileValue(handle->Arg(1)); // high
		op.operands[2]  = CompileValue(inst.Arg(1));    // offset
		if (is_const_buffer_read) {
			op.operands[3] = CompileValue(handle->Arg(2)); // records
			op.operands[4] = CompileValue(handle->Arg(3)); // word3
		}
		return op;
	}

	const ResourcePlan&                       m_program;
	std::vector<CompiledOp>                   m_ops;
	std::unordered_map<const Inst*, uint32_t> m_slot_for_inst;
};

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

float    Float32(uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); }
uint64_t Float32Bits(float value) { return std::bit_cast<uint32_t>(value); }

// One forward pass over `compiled.ops`. Topological order guarantees every operand slot an op
// reads was already produced earlier in this same pass, so this never recurses except for
// ReadFirstLane, which by definition needs a fresh, independently-scoped nested pass (matching
// Evaluator's brand-new-instance-per-ReadFirstLane behaviour -- see SrtCompiler.h).
void ExecutePass(const CompiledSrtProgram& compiled, SrtMemoryReader read_memory, void* userdata,
                 std::span<const uint32_t> user_data, uint64_t shader_base,
                 SrtExecutorScratch& scratch, uint32_t active_mask_slot,
                 const SrtExecutorScratch* clean_scratch) {
	KYTY_PROFILER_FUNCTION();
	const auto count = compiled.ops.size();
	if (scratch.results.size() < count) {
		scratch.results.resize(count);
		scratch.computed.resize(count);
	}
	auto* results  = scratch.results.data();
	auto* computed = scratch.computed.data();
	// One memset instead of a `computed[i] = 0u` store threaded through every loop iteration below
	// -- same total writes, but as a single vectorizable pass instead of interleaved with each op's
	// own branchy case body.
	std::memset(computed, 0, count * sizeof(*computed));

	const auto operand_ok = [&](const CompiledOp& op, uint32_t index) {
		const auto slot = op.operands[index];
		return slot != kInvalidSlot && computed[slot] != 0u;
	};

	for (uint32_t i = 0; i < count; i++) {
		const auto& op = compiled.ops[i];
		switch (op.kind) {
			case CompiledOpKind::AlwaysFails: break;
			case CompiledOpKind::Constant:
				results[i]  = op.immediate;
				computed[i] = 1u;
				break;
			case CompiledOpKind::GetUserData:
				if (op.immediate < user_data.size()) {
					results[i]  = user_data[op.immediate];
					computed[i] = 1u;
				}
				break;
			case CompiledOpKind::GetShaderBase:
				results[i]  = shader_base;
				computed[i] = 1u;
				break;
			case CompiledOpKind::ReadFirstLane: {
				if (!operand_ok(op, 0) || !operand_ok(op, 1)) {
					break;
				}
				// This nested pass must not reuse (or pollute) the outer pass's cache, since the
				// active-mask shortcut below can make the same slot evaluate differently
				// depending on which mask scope is active -- each nesting depth needs its own,
				// independent storage (matching Evaluator's brand-new-instance-per-ReadFirstLane
				// behaviour). A bare thread_local buffer isn't safe here: a ReadFirstLane whose
				// own mask operand is itself gated by another ReadFirstLane recurses into this
				// same case, and the inner call's reset would clobber the outer call's in-flight
				// state. Instead, borrow slot g_nested_scratch_depth from a thread_local pool
				// indexed by nesting depth -- each depth gets its own buffer, reused across calls
				// instead of freshly heap-allocated every single time (ReadFirstLane is common
				// enough in SRT graphs that a fresh pair of vector allocations per hit was
				// measurable). std::deque, not std::vector: growing it while this frame still
				// holds `nested_scratch` as a reference must not invalidate that reference, which
				// vector's reallocation-on-growth would risk if a deeper nested call grows the
				// pool while this frame is still using its own slot.
				thread_local std::deque<SrtExecutorScratch> g_nested_scratch_pool;
				thread_local uint32_t                       g_nested_scratch_depth = 0;
				if (g_nested_scratch_pool.size() <= g_nested_scratch_depth) {
					g_nested_scratch_pool.emplace_back();
				}
				auto& nested_scratch = g_nested_scratch_pool[g_nested_scratch_depth];
				++g_nested_scratch_depth;
				ExecutePass(compiled, read_memory, userdata, user_data, shader_base, nested_scratch,
				           op.operands[1], clean_scratch);
				--g_nested_scratch_depth;
				const auto value_slot = op.operands[0];
				if (nested_scratch.computed[value_slot] != 0u) {
					results[i]  = nested_scratch.results[value_slot];
					computed[i] = 1u;
				}
				break;
			}
			case CompiledOpKind::ReadConst: {
				const auto* source_results  = op.srt_slot_clean && clean_scratch != nullptr
				                                  ? clean_scratch->results.data()
				                                  : results;
				const auto* source_computed = op.srt_slot_clean && clean_scratch != nullptr
				                                   ? clean_scratch->computed.data()
				                                   : computed;
				const auto  slot            = op.operands[0];
				const auto  available = op.srt_slot_clean && clean_scratch != nullptr
				                             ? slot < clean_scratch->computed.size()
				                             : true;
				if (available && source_computed[slot] != 0u) {
					results[i]  = source_results[slot];
					computed[i] = 1u;
				}
				break;
			}
			case CompiledOpKind::ExtractU64: {
				if (!operand_ok(op, 0)) {
					break;
				}
				const auto packed = results[op.operands[0]];
				results[i]        = static_cast<uint32_t>(packed >> (op.component * 32u));
				computed[i]       = 1u;
				break;
			}
			case CompiledOpKind::ExtractPassthrough: {
				if (!operand_ok(op, 0)) {
					break;
				}
				results[i]  = results[op.operands[0]];
				computed[i] = 1u;
				break;
			}
			case CompiledOpKind::ExtractCarryHalf: {
				if (!operand_ok(op, 0) || !operand_ok(op, 1)) {
					break;
				}
				const auto lhs = results[op.operands[0]];
				const auto rhs = results[op.operands[1]];
				const auto sum =
				    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
				results[i] = op.component == 0u ? static_cast<uint32_t>(sum)
				                                 : static_cast<uint32_t>(sum >> 32u);
				computed[i] = 1u;
				break;
			}
			case CompiledOpKind::RawRead: {
				if (!operand_ok(op, 0) || !operand_ok(op, 1) || !operand_ok(op, 2)) {
					break;
				}
				const auto low    = results[op.operands[0]];
				const auto high   = results[op.operands[1]];
				const auto offset = results[op.operands[2]];
				const auto base   = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
				const auto immediate = static_cast<int64_t>(op.immediate);
				uint64_t   address   = 0;
				if (op.is_const_buffer_read) {
					if (!operand_ok(op, 3) || !operand_ok(op, 4)) {
						break;
					}
					const auto records = results[op.operands[3]];
					if (immediate < 0) {
						break;
					}
					const auto byte_offset =
					    static_cast<uint64_t>(immediate) + static_cast<uint32_t>(offset);
					const auto aligned = byte_offset & ~uint64_t {3};
					const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
					const auto size    = stride == 0u
					                         ? static_cast<uint64_t>(static_cast<uint32_t>(records))
					                         : static_cast<uint64_t>(stride) *
					                               static_cast<uint32_t>(records);
					if (aligned > size || size - aligned < sizeof(uint32_t)) {
						break;
					}
					address = ((base & ~uint64_t {3}) + byte_offset) & ~uint64_t {3};
				} else {
					const auto relative = (immediate & ~int64_t {3}) +
					                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
					if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
						break;
					}
				}
				uint32_t word = 0;
				if (read_memory != nullptr) {
					if (!read_memory(userdata, address, &word)) {
						break;
					}
				} else {
					std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
				}
				results[i]  = word;
				computed[i] = 1u;
				break;
			}
			case CompiledOpKind::Select: {
				// Active-mask shortcut: matches EvaluateWide's `inst->Arg(0).Resolve() ==
				// m_active_mask` check -- here, "the same Inst*" is "the same compiled slot". If
				// this Select's own predicate is the ReadFirstLane mask currently in scope, the
				// lane(s) being read are definitionally inside that mask, so the predicate must
				// be true for them -- take the true branch without evaluating the predicate.
				if (active_mask_slot != kInvalidSlot && op.operands[0] == active_mask_slot) {
					if (!operand_ok(op, 1)) {
						break;
					}
					results[i]  = results[op.operands[1]];
					computed[i] = 1u;
					break;
				}
				// Otherwise mirrors EvaluateInst's Select case: the predicate is resolved
				// against the clean pass (when one is available) rather than this pass's own
				// results -- matching `m_clean_evaluator != nullptr ? *m_clean_evaluator :
				// *this` -- and only the branch the predicate actually selects is required; the
				// other is never touched, exactly like the interpreter's lazy Arg() call on only
				// one branch.
				const auto* pred_results  = clean_scratch != nullptr
				                                ? clean_scratch->results.data()
				                                : results;
				const auto* pred_computed = clean_scratch != nullptr
				                                 ? clean_scratch->computed.data()
				                                 : computed;
				const auto  pred_slot = op.operands[0];
				const auto  pred_available =
				    clean_scratch != nullptr ? pred_slot < clean_scratch->computed.size() : true;
				if (!pred_available || pred_computed[pred_slot] == 0u) {
					break;
				}
				const auto chosen = op.operands[pred_results[pred_slot] != 0u ? 1u : 2u];
				if (chosen == kInvalidSlot || computed[chosen] == 0u) {
					break;
				}
				results[i]  = results[chosen];
				computed[i] = 1u;
				break;
			}
			case CompiledOpKind::Generic: {
				bool ok = true;
				for (uint8_t a = 0; a < op.num_operands && ok; a++) {
					ok = operand_ok(op, a);
				}
				if (!ok) {
					break;
				}
				const auto a = results[op.operands[0]];
				const auto b = op.num_operands > 1 ? results[op.operands[1]] : 0;
				const auto c = op.num_operands > 2 ? results[op.operands[2]] : 0;
				const auto d = op.num_operands > 3 ? results[op.operands[3]] : 0;
				uint64_t   result = 0;
				bool       valid  = true;
				switch (op.opcode) {
					case ValueOpcode::BitCastU32F32:
					case ValueOpcode::BitCastF32U32: result = a; break;
					case ValueOpcode::CompositeConstructU64:
						result = static_cast<uint32_t>(a) |
						         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
						break;
					case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); break;
					case ValueOpcode::IAdd64: result = a + b; break;
					case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); break;
					case ValueOpcode::ISub64: result = a - b; break;
					case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); break;
					case ValueOpcode::IMul64: result = a * b; break;
					case ValueOpcode::UMin32:
						result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
						break;
					case ValueOpcode::ConvertF32U32:
						result = Float32Bits(static_cast<float>(static_cast<uint32_t>(a)));
						break;
					case ValueOpcode::ConvertU32F32: {
						const auto value = Float32(a);
						if (!std::isfinite(value) || value < 0.0f ||
						    static_cast<double>(value) > UINT32_MAX) {
							valid = false;
							break;
						}
						result = static_cast<uint32_t>(value);
						break;
					}
					case ValueOpcode::FPMul32: result = Float32Bits(Float32(a) * Float32(b)); break;
					case ValueOpcode::FPTrunc32: result = Float32Bits(std::trunc(Float32(a))); break;
					case ValueOpcode::FPIsNan32: result = std::isnan(Float32(a)); break;
					case ValueOpcode::FPOrdLessThanEqual32: result = Float32(a) <= Float32(b); break;
					case ValueOpcode::FPOrdGreaterThanEqual32:
						result = Float32(a) >= Float32(b);
						break;
					case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); break;
					case ValueOpcode::BitwiseAnd64: result = a & b; break;
					case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); break;
					case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); break;
					case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); break;
					case ValueOpcode::ShiftLeftLogical32:
						result = static_cast<uint32_t>(a) << (b & 31u);
						break;
					case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); break;
					case ValueOpcode::ShiftRightLogical32:
						result = static_cast<uint32_t>(a) >> (b & 31u);
						break;
					case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); break;
					case ValueOpcode::ShiftRightArithmetic32:
						result = static_cast<uint32_t>(
						    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
						break;
					case ValueOpcode::ShiftRightArithmetic64:
						result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
						break;
					case ValueOpcode::BitFieldUExtract: {
						const auto offset = static_cast<uint32_t>(b);
						const auto width  = static_cast<uint32_t>(c);
						if (offset > 32u || width > 32u - offset) {
							valid = false;
							break;
						}
						const auto mask = width == 32u  ? UINT32_MAX
						                  : width == 0u ? 0u
						                                : (uint32_t {1} << width) - 1u;
						result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
						break;
					}
					case ValueOpcode::BitFieldSExtract: {
						const auto offset = static_cast<uint32_t>(b);
						const auto width  = static_cast<uint32_t>(c);
						if (offset > 32u || width > 32u - offset) {
							valid = false;
							break;
						}
						if (width == 0u) {
							result = 0;
							break;
						}
						const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
						auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
						if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
							bits |= ~mask;
						}
						result = bits;
						break;
					}
					case ValueOpcode::BitFieldInsert: {
						const auto offset = static_cast<uint32_t>(c);
						const auto width  = static_cast<uint32_t>(d);
						if (offset > 32u || width > 32u - offset) {
							valid = false;
							break;
						}
						if (width == 0u) {
							result = static_cast<uint32_t>(a);
							break;
						}
						const auto mask =
						    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
						result = (static_cast<uint32_t>(a) & ~mask) |
						         ((static_cast<uint32_t>(b) << offset) & mask);
						break;
					}
					case ValueOpcode::IEqual32:
						result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
						break;
					case ValueOpcode::INotEqual32:
						result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
						break;
					case ValueOpcode::ULessThan32:
						result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
						break;
					case ValueOpcode::UGreaterThan32:
						result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
						break;
					case ValueOpcode::LogicalAnd: result = (a != 0u) && (b != 0u); break;
					case ValueOpcode::LogicalOr: result = (a != 0u) || (b != 0u); break;
					case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); break;
					case ValueOpcode::LogicalNot: result = a == 0u; break;
					default: valid = false; break;
				}
				if (valid) {
					results[i]  = result;
					computed[i] = 1u;
				}
				break;
			}
		}
	}
}

} // namespace

CompiledSrtProgram CompileSrtProgram(const ResourcePlan& program) {
	return Compiler(program).Run();
}

void ExecuteSrtProgram(const CompiledSrtProgram& compiled, SrtMemoryReader read_memory,
                       void* userdata, std::span<const uint32_t> user_data, uint64_t shader_base,
                       SrtExecutorScratch& scratch, uint32_t active_mask_slot,
                       const SrtExecutorScratch* clean_scratch) {
	ExecutePass(compiled, read_memory, userdata, user_data, shader_base, scratch, active_mask_slot,
	           clean_scratch);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
