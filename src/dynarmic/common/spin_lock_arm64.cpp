/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

// Vita3K M12.7.2 hotfix: the upstream implementation placed
// `SpinLockImpl impl;` at namespace scope, meaning its constructor (and
// therefore `oaknut::CodeBlock{4096}`) executed at dlopen time as part
// of the dylib's global initialisers.  On an iOS app without the JIT
// entitlement / a pre-iOS-26 host, that allocation raises std::bad_alloc
// and crashes the process before the libretro frontend even gets a
// chance to read retro_set_environment().
//
// Fixes in this file:
//   1. `mem` / `code` are now `std::unique_ptr`, so the global ctor is
//      free and no JIT memory is requested at dlopen.
//   2. `SpinLockImpl::Initialize()` is guarded by `std::call_once`
//      (as before) and wrapped in try/catch — on alloc failure we
//      fall back to portable C atomic intrinsics (`__atomic_*`) so
//      ExclusiveMonitor::Lock/Unlock keep working even when picked
//      through the IR interpreter backend on iOS <26 devices.
//   3. The C fallback spins with `yield` on arm64 / `pause` on x64,
//      matching what the JIT blob would have done with WFE.

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>

#if defined(__APPLE__)
#    include <TargetConditionals.h>
#endif

#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/common/spin_lock.h"

namespace Dynarmic {

using Backend::Arm64::Wscratch0;
using Backend::Arm64::Wscratch1;
using namespace oaknut::util;

void EmitSpinLockLock(oaknut::CodeGenerator& code, oaknut::XReg ptr) {
    oaknut::Label start, loop;

    code.MOV(Wscratch1, 1);
    code.SEVL();
    code.l(start);
    code.WFE();
    code.l(loop);
    code.LDAXR(Wscratch0, ptr);
    code.CBNZ(Wscratch0, start);
    code.STXR(Wscratch0, Wscratch1, ptr);
    code.CBNZ(Wscratch0, loop);
}

void EmitSpinLockUnlock(oaknut::CodeGenerator& code, oaknut::XReg ptr) {
    code.STLR(WZR, ptr);
}

namespace {

using LockFn = void (*)(volatile int*);

// Portable C fallback used when JIT code allocation is unavailable.
// Semantically equivalent to the hand-written WFE spin loop, only slower
// on contended hot paths.  Safe on arm64, x86_64 and any other target
// exposing GCC atomic builtins.
void c_spin_lock(volatile int* s) {
    int expected = 0;
    while (!__atomic_compare_exchange_n(s, &expected, 1, /*weak*/ true,
                                        __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        expected = 0;
#if defined(__aarch64__)
        __asm__ volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
        __asm__ volatile("pause");
#endif
    }
}

void c_spin_unlock(volatile int* s) {
    __atomic_store_n(s, 0, __ATOMIC_RELEASE);
}

struct SpinLockImpl {
    // Cheap default ctor -- no JIT allocation, safe to run during dlopen.
    SpinLockImpl() = default;

    // Called once from std::call_once at first Lock/Unlock invocation.
    // Tries the JIT path; on failure leaves `lock`/`unlock` bound to the
    // already-initialised C fallback.
    void Initialize();

    std::unique_ptr<oaknut::CodeBlock> mem;
    std::unique_ptr<oaknut::CodeGenerator> code;

    // Default-initialise to the C fallback so that Lock/Unlock are safe
    // even if the very first call races with Initialize() (call_once
    // serialises us there, but belt-and-braces).
    LockFn lock   = &c_spin_lock;
    LockFn unlock = &c_spin_unlock;
};

std::once_flag flag;
SpinLockImpl impl;

void SpinLockImpl::Initialize() {
    // Try oaknut JIT-backed spin helpers on all iOS arm64 hosts: iOS <26 uses
    // Legacy W^X inside oaknut::CodeBlock; iOS 26+ uses TXM or PPL dual maps.
    // On failure we keep the portable C atomics (interpreter-safe).
    try {
        mem  = std::make_unique<oaknut::CodeBlock>(4096);
        code = std::make_unique<oaknut::CodeGenerator>(mem->wptr(), mem->xptr());
    } catch (const std::bad_alloc&) {
        // JIT unavailable (alloc failed on a device we thought was TXM-
        // capable).  Keep the C fallback and stay silent.
        mem.reset();
        code.reset();
        return;
    }

    mem->unprotect();

    lock = code->xptr<LockFn>();
    EmitSpinLockLock(*code, X0);
    code->RET();

    unlock = code->xptr<LockFn>();
    EmitSpinLockUnlock(*code, X0);
    code->RET();

    mem->protect();
    mem->invalidate_all();
}

}  // namespace

void SpinLock::Lock() {
    std::call_once(flag, &SpinLockImpl::Initialize, impl);
    impl.lock(&storage);
}

void SpinLock::Unlock() {
    std::call_once(flag, &SpinLockImpl::Initialize, impl);
    impl.unlock(&storage);
}

}  // namespace Dynarmic
