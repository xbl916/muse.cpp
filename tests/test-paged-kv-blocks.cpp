#include "llama-kv-cache-paged.h"

#include <cstdio>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int test_allocator() {
    llama_kv_block_allocator alloc;
    alloc.init(96, 32);

    REQUIRE(alloc.n_blocks() == 3);
    REQUIRE(alloc.n_free() == 3);
    REQUIRE(alloc.alloc() == 0);
    REQUIRE(alloc.alloc() == 1);
    REQUIRE(alloc.n_free() == 1);
    REQUIRE(alloc.retain(0));
    REQUIRE(alloc.ref_count(0) == 2);
    REQUIRE(alloc.release(0));
    REQUIRE(alloc.ref_count(0) == 1);
    REQUIRE(alloc.release(0));
    REQUIRE(alloc.ref_count(0) == 0);
    REQUIRE(alloc.n_free() == 2);
    REQUIRE(alloc.alloc() == 0);
    REQUIRE(alloc.alloc() == 2);
    REQUIRE(alloc.alloc() == LLAMA_KV_BLOCK_ID_NONE);

    alloc.reset();
    REQUIRE(alloc.n_free() == 3);
    REQUIRE(alloc.acquire(2));
    REQUIRE(alloc.ref_count(2) == 1);
    REQUIRE(alloc.n_free() == 2);

    return 0;
}

static int test_table() {
    llama_kv_block_table table;
    table.init(2);

    REQUIRE(table.insert(0, 0, 4));
    REQUIRE(table.insert(0, 2, 7));
    REQUIRE(table.lookup(0, 0) == 4);
    REQUIRE(table.lookup(0, 1) == LLAMA_KV_BLOCK_ID_NONE);
    REQUIRE(table.lookup(0, 2) == 7);
    REQUIRE(table.lookup(2, 0) == LLAMA_KV_BLOCK_ID_NONE);
    REQUIRE(table.n_tokens(0) == 0);
    REQUIRE(table.set_n_tokens(0, 513));
    REQUIRE(table.n_tokens(0) == 513);
    REQUIRE(table.size() == 2);
    REQUIRE(table.erase_page(0, 0) == 4);
    REQUIRE(table.size() == 1);

    const auto released = table.erase_seq(0);
    REQUIRE(released.size() == 1);
    REQUIRE(released[0] == 7);
    REQUIRE(table.size() == 0);
    REQUIRE(table.n_tokens(0) == 0);
    REQUIRE(llama_kv_block_table::logical_page(63, 32) == 1);
    REQUIRE(llama_kv_block_table::intra_offset(63, 32) == 31);

    return 0;
}

int main() {
    if (test_allocator() != 0) {
        return 1;
    }
    return test_table();
}
