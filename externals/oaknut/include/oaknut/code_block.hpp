// SPDX-FileCopyrightText: Copyright (c) 2022 merryhime <https://mary.rs>
// SPDX-License-Identifier: MIT
//
// Vita3K: iOS JIT paths merged from Azahar (TXM / iOS26+PPL / Legacy <26) inside
// oaknut::CodeBlock — no separate vita3k_ios_jit_alloc bridge. Dynarmic uses
// ptr()/wptr(); Vita3K adds xptr() as an alias of ptr() for the arm64 backend.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <cerrno>

#if defined(_WIN32)
#    define NOMINMAX
#    include <windows.h>
#elif defined(__APPLE__)
#    include <TargetConditionals.h>
#    include <libkern/OSCacheControl.h>
#    include <pthread.h>
#    include <sys/mman.h>
#    include <sys/sysctl.h>
#    include <unistd.h>
#    include <mach/mach.h>
#    include <mach/vm_map.h>
#    include <dlfcn.h>
#    include <dirent.h>
#    include <cstring>

// Mach constants for iOS 26 JIT
#ifndef MAP_MEM_NAMED_CREATE
#    define MAP_MEM_NAMED_CREATE 0x020000
#endif
#ifndef MAP_MEM_LEDGER_TAGGED
#    define MAP_MEM_LEDGER_TAGGED 0x002000
#endif
#ifndef VM_LEDGER_TAG_DEFAULT
#    define VM_LEDGER_TAG_DEFAULT 0x00000001
#endif
#ifndef VM_LEDGER_FLAG_NO_FOOTPRINT
#    define VM_LEDGER_FLAG_NO_FOOTPRINT 0x00000001
#endif

// mach_memory_entry_ownership may not be declared in all SDK versions
extern "C" {
    kern_return_t mach_memory_entry_ownership(
        mem_entry_name_port_t mem_entry,
        mach_port_t owner,
        int ledger_tag,
        int ledger_flags) __attribute__((weak_import));
}
#else
#    include <sys/mman.h>
#endif

namespace oaknut {

// 检测是否为iOS平台 - 同时支持SDK宏和构建系统宏
#if defined(__APPLE__) && defined(__arm64__)
#  if TARGET_OS_IPHONE || defined(IOS)
#    define OAKNUT_IOS_ARM64 1
#  endif
#endif

#ifdef __APPLE__
#include <os/log.h>
static os_log_t oaknut_log = os_log_create("com.vita3k.oaknut", "CodeBlock");
#endif

#ifdef OAKNUT_IOS_ARM64

namespace detail {
// Set after the first successful StikDebug TXM brk #0xf00d handshake for this process.
inline std::atomic<bool> g_vita3k_txm_handshake_done{false};
} // namespace detail

// CS_OPS 常量
#define OAKNUT_CS_OPS_STATUS 0
#define OAKNUT_CS_DEBUGGED 0x10000000

extern "C" int csops(pid_t pid, unsigned int ops, void* useraddr, size_t usersize);

// Check if process is being debugged
inline bool CodeBlockIsProcessDebugged() {
    int flags = 0;
    if (csops(getpid(), OAKNUT_CS_OPS_STATUS, &flags, sizeof(flags)) != 0) {
        os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] csops call failed");
        return false;
    }
    bool debugged = (flags & OAKNUT_CS_DEBUGGED) != 0;
    os_log(oaknut_log, "%{public}s %{public}s (flags=0x%08X)", "[>>oaknut CodeBlock] Debug status:", debugged ? "attached" : "not attached", flags);
    return debugged;
}

// Get major version number
inline int CodeBlockGetIOSMajorVersion() {
    char version_str[256] = {0};
    size_t size = sizeof(version_str);
    
    if (sysctlbyname("kern.osproductversion", version_str, &size, nullptr, 0) == 0) {
        int major = std::atoi(version_str);
        os_log(oaknut_log, "%{public}s %{public}s (major: %d)", "[>>oaknut CodeBlock] System version:", version_str, major);
        return major;
    }
    return 0;
}

// Check if iOS version is 26 or later
inline bool CodeBlockIsIOS26OrLater() {
    static bool checked = false;
    static bool is_ios26 = false;
    
    if (checked) return is_ios26;
    checked = true;
    
#if __has_builtin(__builtin_available)
    if (__builtin_available(iOS 26, *)) {
        is_ios26 = true;
        os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] Version check: iOS 26+ (builtin)");
        return true;
    }
#endif
    
    int major_version = CodeBlockGetIOSMajorVersion();
    if (major_version >= 26) {
        is_ios26 = true;
        os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] Version check: iOS 26+ (sysctl)");
        return true;
    }
    
    os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] Version check: iOS < 26");
    return false;
}

// Helper function: find entry with specified length in path
inline bool CodeBlockFindPathWithLength(const char* base_path, size_t target_length, char* out_path, size_t out_size) {
    DIR* dir = opendir(base_path);
    if (!dir) return false;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strlen(entry->d_name) == target_length) {
            snprintf(out_path, out_size, "%s/%s", base_path, entry->d_name);
            closedir(dir);
            return true;
        }
    }
    closedir(dir);
    return false;
}

// Check if device uses TXM
inline bool CodeBlockDeviceHasTXM() {
    static bool checked = false;
    static bool has_txm = false;
    
    if (checked) return has_txm;
    checked = true;
    
    os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] Detecting TXM...");
    
    // Method 1
    char boot_uuid_path[512];
    if (CodeBlockFindPathWithLength("/System/Volumes/Preboot", 36, boot_uuid_path, sizeof(boot_uuid_path))) {
        char boot_dir[512];
        snprintf(boot_dir, sizeof(boot_dir), "%s/boot", boot_uuid_path);
        
        char ninety_six_path[512];
        if (CodeBlockFindPathWithLength(boot_dir, 96, ninety_six_path, sizeof(ninety_six_path))) {
            char txm_path[1024];
            snprintf(txm_path, sizeof(txm_path), 
                     "%s/usr/standalone/firmware/FUD/Ap,TrustedExecutionMonitor.img4", 
                     ninety_six_path);
            
            if (access(txm_path, F_OK) == 0) {
                has_txm = true;
                os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] TXM detection: TXM found");
                return true;
            }
        }
    }
    
    // Method 2
    char fallback_path[512];
    if (CodeBlockFindPathWithLength("/private/preboot", 96, fallback_path, sizeof(fallback_path))) {
        char txm_path[1024];
        snprintf(txm_path, sizeof(txm_path), 
                 "%s/usr/standalone/firmware/FUD/Ap,TrustedExecutionMonitor.img4", 
                 fallback_path);
        
        if (access(txm_path, F_OK) == 0) {
            has_txm = true;
            os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] TXM detection: TXM found");
            return true;
        }
    }
    
    os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] TXM detection: Device does not use TXM (PPL)");
    return false;
}

// Get system page size
inline std::size_t CodeBlockGetPageSize() {
    static std::size_t page_size = 0;
    if (page_size == 0) {
        page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        if (page_size == 0) page_size = 16384; // Default 16KB for iOS
    }
    return page_size;
}

// Align to page size
inline std::size_t CodeBlockAlignToPage(std::size_t size) {
    std::size_t page_size = CodeBlockGetPageSize();
    return (size + page_size - 1) & ~(page_size - 1);
}

#endif // OAKNUT_IOS_ARM64

class CodeBlock {
public:
    explicit CodeBlock(std::size_t size)
        : m_size(size)
    {
#ifdef __APPLE__
        os_log(oaknut_log, "%{public}s size=%zu", "[>>oaknut CodeBlock] Constructor started,", size);
#endif
        
#if defined(_WIN32)
        m_memory = (std::uint32_t*)VirtualAlloc(nullptr, size, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#elif defined(__APPLE__)

#ifdef OAKNUT_IOS_ARM64
        os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] iOS ARM64: OAKNUT_IOS_ARM64 defined");
        
        bool is_ios26 = CodeBlockIsIOS26OrLater();
        bool has_txm = false;
        
        if (is_ios26) {
            has_txm = CodeBlockDeviceHasTXM();
        }
        
        // iOS 26 TXM mode: use dual mapping, rx for execution, rw for writing
        const bool use_txm_for_this_block = is_ios26 && has_txm;
        // iOS 26 NoTXM/PPL mode: also use dual mapping
        const bool use_no_txm_for_this_block = is_ios26 && !has_txm;
        
        if (use_txm_for_this_block) {
            // iOS 26+ TXM device - use mach_make_memory_entry_64 method
            // Key improvement: align request size to page size to avoid 4096 byte issues
            std::size_t page_size = CodeBlockGetPageSize();
            std::size_t aligned_size = CodeBlockAlignToPage(size);
            
            os_log(oaknut_log, "%{public}s requested=%zu, page_aligned=%zu, page_size=%zu", 
                   "[>>oaknut CodeBlock] TXM mode:", size, aligned_size, page_size);
            
            // Step 1: Create memory entry
            memory_object_size_t memory_size = aligned_size;
            mach_port_t memory_entry = MACH_PORT_NULL;
            
            kern_return_t ret = mach_make_memory_entry_64(
                mach_task_self(),
                &memory_size,
                0,
                MAP_MEM_NAMED_CREATE | MAP_MEM_LEDGER_TAGGED | 
                VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE,
                &memory_entry,
                MACH_PORT_NULL
            );
            
            if (ret != KERN_SUCCESS || memory_size < aligned_size) {
                os_log(oaknut_log, "%{public}s ret=0x%x, memory_size=%llu, expected=%zu", 
                       "[>>oaknut CodeBlock] TXM: [FATAL] mach_make_memory_entry_64 failed,",
                       ret, (unsigned long long)memory_size, aligned_size);
                if (memory_entry != MACH_PORT_NULL) {
                    mach_port_deallocate(mach_task_self(), memory_entry);
                }
                throw std::bad_alloc{};
            }
            
            std::size_t actual_size = static_cast<std::size_t>(memory_size);
            os_log(oaknut_log, "%{public}s actual_size=%zu", "[>>oaknut CodeBlock] TXM: mach_make_memory_entry_64 succeeded,", actual_size);
            
            // Step 2: Set ownership to avoid memory footprint (optional)
            if (mach_memory_entry_ownership != nullptr) {
                ret = mach_memory_entry_ownership(
                    memory_entry,
                    MACH_PORT_NULL,
                    VM_LEDGER_TAG_DEFAULT,
                    VM_LEDGER_FLAG_NO_FOOTPRINT
                );
                if (ret != KERN_SUCCESS) {
                    os_log(oaknut_log, "%{public}s ret=0x%x", "[>>oaknut CodeBlock] TXM: mach_memory_entry_ownership failed (non-fatal),", ret);
                }
            }
            
            // Step 3: Map rx region
            vm_address_t rx_addr = 0;
            ret = vm_map(
                mach_task_self(),
                &rx_addr,
                actual_size,
                0,
                VM_FLAGS_ANYWHERE,
                memory_entry,
                0,
                FALSE,
                VM_PROT_READ | VM_PROT_EXECUTE,
                VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE,
                VM_INHERIT_COPY
            );
            
            // Release memory_entry port (regardless of success)
            mach_port_deallocate(mach_task_self(), memory_entry);
            
            if (ret != KERN_SUCCESS) {
                os_log(oaknut_log, "%{public}s ret=0x%x", "[>>oaknut CodeBlock] TXM: [FATAL] vm_map rx failed,", ret);
                throw std::bad_alloc{};
            }
            
            os_log(oaknut_log, "%{public}s addr=0x%llx, size=%zu", "[>>oaknut CodeBlock] TXM: vm_map rx succeeded:", (unsigned long long)rx_addr, actual_size);
            
            // Execute brk #0xf00d to notify StikDebug to mark memory as executable
            const char* skip_brk = getenv("VITA3K_SKIP_BRK");
            if (skip_brk && std::atoi(skip_brk) != 0) {
                os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] TXM: Skipping brk #0xf00d (VITA3K_SKIP_BRK is set)");
            } else {
                os_log(oaknut_log, "%{public}s addr=0x%llx, size=%zu", "[>>oaknut CodeBlock] TXM: Executing brk #0xf00d,", (unsigned long long)rx_addr, actual_size);
                __asm__ volatile (
                    "mov x0, %0\n"
                    "mov x1, %1\n"
                    "mov x16, #1\n"
                    "brk #0xf00d"
                    :
                    : "r" (rx_addr), "r" (actual_size)
                    : "x0", "x1", "x16"
                );
                detail::g_vita3k_txm_handshake_done.store(true, std::memory_order_release);
                os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] TXM: brk #0xf00d completed");
            }
            
            // Step 4: Use vm_remap to create rw mirror
            vm_address_t rw_addr = 0;
            vm_prot_t cur_prot = 0, max_prot = 0;
            
            ret = vm_remap(
                mach_task_self(),
                &rw_addr,
                actual_size,
                0,
                VM_FLAGS_ANYWHERE,
                mach_task_self(),
                rx_addr,
                FALSE,
                &cur_prot,
                &max_prot,
                VM_INHERIT_NONE
            );
            
            if (ret != KERN_SUCCESS) {
                os_log(oaknut_log, "%{public}s ret=0x%x", "[>>oaknut CodeBlock] TXM: [FATAL] vm_remap failed,", ret);
                vm_deallocate(mach_task_self(), rx_addr, actual_size);
                throw std::bad_alloc{};
            }
            
            os_log(oaknut_log, "%{public}s rw=0x%llx, cur_prot=0x%x, max_prot=0x%x", 
                   "[>>oaknut CodeBlock] TXM: vm_remap succeeded:", (unsigned long long)rw_addr, cur_prot, max_prot);
            
            // Step 5: Set rw permissions
            if (mprotect((void*)rw_addr, actual_size, PROT_READ | PROT_WRITE) != 0) {
                os_log(oaknut_log, "%{public}s errno=%d", "[>>oaknut CodeBlock] TXM: [FATAL] mprotect rw failed,", errno);
                vm_deallocate(mach_task_self(), rw_addr, actual_size);
                vm_deallocate(mach_task_self(), rx_addr, actual_size);
                throw std::bad_alloc{};
            }
            
            m_memory = reinterpret_cast<std::uint32_t*>(rx_addr);
            m_rw_memory = reinterpret_cast<std::uint32_t*>(rw_addr);
            m_is_txm = true;
            m_size = actual_size;
            
            std::ptrdiff_t diff = reinterpret_cast<std::uint8_t*>(m_rw_memory) - reinterpret_cast<std::uint8_t*>(m_memory);
            os_log(oaknut_log, "%{public}s rx=0x%llx rw=0x%llx size=%zu diff=%td", 
                   "[>>oaknut CodeBlock] TXM: Init succeeded!", (unsigned long long)m_memory, (unsigned long long)m_rw_memory, actual_size, diff);
            return;
        }
        
        if (use_no_txm_for_this_block) {
            // iOS 26+ PPL device - use dual mapping mode
            std::size_t aligned_size = CodeBlockAlignToPage(size);
            os_log(oaknut_log, "%{public}s requested=%zu, page_aligned=%zu", "[>>oaknut CodeBlock] NoTXM/PPL mode:", size, aligned_size);
            
            void* rx_ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
            if (rx_ptr == MAP_FAILED) {
                os_log(oaknut_log, "%{public}s errno=%d", "[>>oaknut CodeBlock] NoTXM: mmap r-x failed,", errno);
                throw std::bad_alloc{};
            }
            
            os_log(oaknut_log, "%{public}s addr=0x%llx, size=%zu", "[>>oaknut CodeBlock] NoTXM: mmap r-x succeeded:", (unsigned long long)rx_ptr, aligned_size);
            
            // Create rw mirror
            vm_address_t rw_addr = 0;
            vm_prot_t cur_prot = 0, max_prot = 0;
            
            kern_return_t ret = vm_remap(
                mach_task_self(),
                &rw_addr,
                aligned_size,
                0,
                VM_FLAGS_ANYWHERE,
                mach_task_self(),
                (vm_address_t)rx_ptr,
                FALSE,
                &cur_prot,
                &max_prot,
                VM_INHERIT_NONE
            );
            
            if (ret != KERN_SUCCESS) {
                os_log(oaknut_log, "%{public}s ret=0x%x", "[>>oaknut CodeBlock] NoTXM: vm_remap failed,", ret);
                munmap(rx_ptr, aligned_size);
                throw std::bad_alloc{};
            }
            
            if (mprotect((void*)rw_addr, aligned_size, PROT_READ | PROT_WRITE) != 0) {
                os_log(oaknut_log, "%{public}s errno=%d", "[>>oaknut CodeBlock] NoTXM: mprotect failed,", errno);
                vm_deallocate(mach_task_self(), rw_addr, aligned_size);
                munmap(rx_ptr, aligned_size);
                throw std::bad_alloc{};
            }
            
            m_memory = reinterpret_cast<std::uint32_t*>(rx_ptr);
            m_rw_memory = reinterpret_cast<std::uint32_t*>(rw_addr);
            m_is_no_txm = true;
            m_size = aligned_size;
            
            os_log(oaknut_log, "%{public}s rx=0x%llx rw=0x%llx size=%zu", 
                   "[>>oaknut CodeBlock] NoTXM: Init succeeded!", (unsigned long long)m_memory, (unsigned long long)m_rw_memory, aligned_size);
            return;
        }
        
        // Legacy mode (iOS < 26) - single mapping, use mprotect to switch permissions
        os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] Legacy mode (iOS < 26)");
        m_memory = (std::uint32_t*)mmap(nullptr, size, PROT_READ | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (m_memory == MAP_FAILED) {
            os_log(oaknut_log, "%{public}s errno=%d", "[>>oaknut CodeBlock] Legacy: mmap failed,", errno);
            m_memory = nullptr;
            throw std::bad_alloc{};
        }
        os_log(oaknut_log, "%{public}s rx=0x%llx size=%zu", "[>>oaknut CodeBlock] Legacy: mmap succeeded,", (unsigned long long)m_memory, size);
        // Legacy mode doesn't need m_rw_memory, wptr() will return m_memory
#else
        // macOS or other Apple platforms (OAKNUT_IOS_ARM64 not defined)
        os_log(oaknut_log, "%{public}s", "[>>oaknut CodeBlock] OAKNUT_IOS_ARM64 not defined, using default path");
#    if TARGET_OS_IPHONE
        m_memory = (std::uint32_t*)mmap(nullptr, size, PROT_READ | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
#    else
        m_memory = (std::uint32_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_JIT, -1, 0);
#    endif
#endif // OAKNUT_IOS_ARM64

#elif defined(__NetBSD__)
        m_memory = (std::uint32_t*)mmap(nullptr, size, PROT_MPROTECT(PROT_READ | PROT_WRITE | PROT_EXEC), MAP_ANON | MAP_PRIVATE, -1, 0);
#elif defined(__OpenBSD__)
        m_memory = (std::uint32_t*)mmap(nullptr, size, PROT_READ | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
#else
        m_memory = (std::uint32_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE, -1, 0);
#endif

        if (m_memory == nullptr)
            throw std::bad_alloc{};
    }

    ~CodeBlock()
    {
        if (m_memory == nullptr)
            return;

#if defined(_WIN32)
        VirtualFree((void*)m_memory, 0, MEM_RELEASE);
#elif defined(__APPLE__) && defined(OAKNUT_IOS_ARM64)
        if (m_is_txm) {
            if (m_rw_memory) {
                os_log(oaknut_log, "%{public}s 0x%llx", "[>>oaknut CodeBlock] TXM: Releasing rw mapping", (unsigned long long)m_rw_memory);
                vm_deallocate(mach_task_self(), (vm_address_t)m_rw_memory, m_size);
            }
            os_log(oaknut_log, "%{public}s 0x%llx", "[>>oaknut CodeBlock] TXM: Releasing rx mapping", (unsigned long long)m_memory);
            vm_deallocate(mach_task_self(), (vm_address_t)m_memory, m_size);
            return;
        }
        
        if (m_is_no_txm) {
            if (m_rw_memory) {
                os_log(oaknut_log, "%{public}s 0x%llx", "[>>oaknut CodeBlock] NoTXM: Releasing rw mapping", (unsigned long long)m_rw_memory);
                vm_deallocate(mach_task_self(), (vm_address_t)m_rw_memory, m_size);
            }
            os_log(oaknut_log, "%{public}s 0x%llx", "[>>oaknut CodeBlock] NoTXM: Releasing rx mapping", (unsigned long long)m_memory);
            munmap(m_memory, m_size);
            return;
        }
        
        // Legacy mode (iOS < 26)
        os_log(oaknut_log, "%{public}s 0x%llx", "[>>oaknut CodeBlock] Legacy: Releasing memory", (unsigned long long)m_memory);
        munmap(m_memory, m_size);
#else
        munmap(m_memory, m_size);
#endif
    }

    CodeBlock(const CodeBlock&) = delete;
    CodeBlock& operator=(const CodeBlock&) = delete;
    CodeBlock(CodeBlock&&) = delete;
    CodeBlock& operator=(CodeBlock&&) = delete;

    /// Get executable memory pointer (for executing JIT code and getting function entry points)
    std::uint32_t* ptr() const
    {
        return m_memory;
    }

    /// Vita3K Dynarmic arm64 backend: executable buffer (alias of ptr()).
    std::uint32_t* xptr() const
    {
        return m_memory;
    }

    /// Get writable memory pointer (for writing JIT code)
    /// - iOS 26+ dual mapping mode: returns rw mapping pointer
    /// - iOS < 26 Legacy mode: returns same pointer as ptr() (requires unprotect/protect)
    /// - macOS: returns same pointer as ptr() (with pthread_jit_write_protect_np)
    std::uint32_t* wptr() const
    {
#ifdef OAKNUT_IOS_ARM64
        // iOS 26+ dual mapping mode: return rw mapping
        if (m_rw_memory) {
            return m_rw_memory;
        }
        // Legacy mode: return m_memory (requires unprotect before writing)
#endif
        return m_memory;
    }
    
    /// Check if using dual mapping mode (iOS 26+)
    bool is_dual_mapping() const
    {
#ifdef OAKNUT_IOS_ARM64
        return m_is_txm || m_is_no_txm;
#else
        return false;
#endif
    }

    void protect()
    {
#if defined(__APPLE__) && !defined(OAKNUT_IOS_ARM64)
        pthread_jit_write_protect_np(1);
#elif defined(OAKNUT_IOS_ARM64)
        // iOS 26+ dual mapping mode doesn't need protection switching, rx and rw are separate mappings
        if (m_is_txm || m_is_no_txm) {
            return;
        }
        // Legacy mode (iOS < 26): switch to read-only + execute
        mprotect(m_memory, m_size, PROT_READ | PROT_EXEC);
#elif defined(__APPLE__) || defined(__NetBSD__) || defined(__OpenBSD__)
        mprotect(m_memory, m_size, PROT_READ | PROT_EXEC);
#endif
    }

    void unprotect()
    {
#if defined(__APPLE__) && !defined(OAKNUT_IOS_ARM64)
        pthread_jit_write_protect_np(0);
#elif defined(OAKNUT_IOS_ARM64)
        // iOS 26+ dual mapping mode doesn't need protection switching
        if (m_is_txm || m_is_no_txm) {
            return;
        }
        // Legacy mode (iOS < 26): switch to read-write
        mprotect(m_memory, m_size, PROT_READ | PROT_WRITE);
#elif defined(__APPLE__) || defined(__NetBSD__) || defined(__OpenBSD__)
        mprotect(m_memory, m_size, PROT_READ | PROT_WRITE);
#endif
    }

    void invalidate(std::uint32_t* mem, std::size_t size)
    {
#if defined(__APPLE__)
#ifdef OAKNUT_IOS_ARM64
        // iOS 26+ dual mapping mode: need to use rx pointer for invalidate
        if (m_is_txm || m_is_no_txm) {
            // If passed rw pointer, need to convert to rx pointer
            std::uint8_t* mem_byte = reinterpret_cast<std::uint8_t*>(mem);
            std::uint8_t* rx_base = reinterpret_cast<std::uint8_t*>(m_memory);
            std::uint8_t* rw_base = reinterpret_cast<std::uint8_t*>(m_rw_memory);
            
            // Calculate offset and convert to rx address
            std::uint32_t* invalidate_ptr;
            if (m_rw_memory && mem_byte >= rw_base && mem_byte < rw_base + m_size) {
                // mem is rw address, convert to rx address
                std::ptrdiff_t offset = mem_byte - rw_base;
                invalidate_ptr = reinterpret_cast<std::uint32_t*>(rx_base + offset);
            } else {
                // mem is already rx address
                invalidate_ptr = mem;
            }
            
            sys_icache_invalidate(invalidate_ptr, size);
            return;
        }
#endif
        sys_icache_invalidate(mem, size);
#elif defined(_WIN32)
        FlushInstructionCache(GetCurrentProcess(), mem, size);
#else
        static std::size_t icache_line_size = 0x10000, dcache_line_size = 0x10000;

        std::uint64_t ctr;
        __asm__ volatile("mrs %0, ctr_el0"
                         : "=r"(ctr));

        const std::size_t isize = icache_line_size = std::min<std::size_t>(icache_line_size, 4 << ((ctr >> 0) & 0xf));
        const std::size_t dsize = dcache_line_size = std::min<std::size_t>(dcache_line_size, 4 << ((ctr >> 16) & 0xf));

        const std::uintptr_t end = (std::uintptr_t)mem + size;

        for (std::uintptr_t addr = ((std::uintptr_t)mem) & ~(dsize - 1); addr < end; addr += dsize) {
            __asm__ volatile("dc cvau, %0"
                             :
                             : "r"(addr)
                             : "memory");
        }
        __asm__ volatile("dsb ish\n"
                         :
                         :
                         : "memory");

        for (std::uintptr_t addr = ((std::uintptr_t)mem) & ~(isize - 1); addr < end; addr += isize) {
            __asm__ volatile("ic ivau, %0"
                             :
                             : "r"(addr)
                             : "memory");
        }
        __asm__ volatile("dsb ish\nisb\n"
                         :
                         :
                         : "memory");
#endif
    }

    void invalidate_all()
    {
        invalidate(m_memory, m_size);
    }

protected:
    std::uint32_t* m_memory = nullptr;
#ifdef OAKNUT_IOS_ARM64
    std::uint32_t* m_rw_memory = nullptr;
    bool m_is_txm = false;
    bool m_is_no_txm = false;
#endif
    std::size_t m_size = 0;
};

}  // namespace oaknut
