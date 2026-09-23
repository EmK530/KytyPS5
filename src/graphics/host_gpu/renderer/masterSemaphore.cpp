#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) [[likely]] {
		return;
	}
	Refresh();
	if (IsFree(tick)) [[likely]] {
		return;
	}

	// Fast spin without kernel syscalls
	for (int i = 0; i < 300; ++i) {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#endif
		if ((i & 15) == 0) {
			Refresh();
			if (IsFree(tick)) [[likely]] {
				return;
			}
		}
	}

	KYTY_PROFILER_BLOCK("MasterSemaphore::Wait (blocked on GPU)");

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	// Wait in short 1ms increments to avoid a long kernel sleep; poll/refresh between waits.
	constexpr uint64_t kTimeoutNs = 1'000'000; // 1 ms
	while (!IsFree(tick)) {
		m_graphics.device.waitSemaphores(&wait_info, kTimeoutNs);
		Refresh();
	}

	KYTY_PROFILER_END_BLOCK;
}

} // namespace Libs::Graphics
