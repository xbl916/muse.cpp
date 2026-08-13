#pragma once

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

static constexpr uint32_t LLAMA_KV_BLOCK_ID_NONE = std::numeric_limits<uint32_t>::max();

struct llama_kv_block {
    uint32_t id;
    uint32_t first_cell;
    uint32_t size;

    uint32_t cell(uint32_t offset) const {
        assert(offset < size);
        return first_cell + offset;
    }
};

class llama_kv_block_allocator {
public:
    void init(uint32_t n_cells, uint32_t block_size) {
        assert(block_size > 0 && (block_size & (block_size - 1)) == 0);

        block_size_ = block_size;
        blocks_.clear();
        free_ids_.clear();

        const uint32_t n_blocks = n_cells / block_size;
        blocks_.resize(n_blocks);
        ref_counts_.assign(n_blocks, 0);
        free_ids_.reserve(n_blocks);

        for (uint32_t id = 0; id < n_blocks; ++id) {
            blocks_[id] = { id, id * block_size, block_size };
        }
        for (uint32_t id = n_blocks; id > 0; --id) {
            free_ids_.push_back(id - 1);
        }
    }

    uint32_t n_blocks() const {
        return blocks_.size();
    }

    uint32_t n_free() const {
        return free_ids_.size();
    }

    uint32_t block_size() const {
        return block_size_;
    }

    const llama_kv_block & get(uint32_t id) const {
        assert(id < blocks_.size());
        return blocks_[id];
    }

    uint32_t peek_free(uint32_t offset = 0) const {
        if (offset >= free_ids_.size()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }
        return free_ids_[free_ids_.size() - offset - 1];
    }

    uint32_t alloc() {
        if (free_ids_.empty()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }

        const uint32_t id = free_ids_.back();
        free_ids_.pop_back();
        assert(ref_counts_[id] == 0);
        ref_counts_[id] = 1;
        return id;
    }

    bool acquire(uint32_t id) {
        if (id >= blocks_.size()) {
            return false;
        }

        if (ref_counts_[id] == 0) {
            const auto it = std::find(free_ids_.begin(), free_ids_.end(), id);
            if (it == free_ids_.end()) {
                return false;
            }
            *it = free_ids_.back();
            free_ids_.pop_back();
        }
        ++ref_counts_[id];
        return true;
    }

    bool retain(uint32_t id) {
        if (id >= blocks_.size() || ref_counts_[id] == 0) {
            return false;
        }
        ++ref_counts_[id];
        return true;
    }

    bool release(uint32_t id) {
        if (id >= blocks_.size() || ref_counts_[id] == 0) {
            return false;
        }

        if (--ref_counts_[id] == 0) {
            free_ids_.push_back(id);
        }
        return true;
    }

    uint32_t ref_count(uint32_t id) const {
        assert(id < ref_counts_.size());
        return ref_counts_[id];
    }

    bool is_shared(uint32_t id) const {
        return ref_count(id) > 1;
    }

    void reset() {
        free_ids_.clear();
        std::fill(ref_counts_.begin(), ref_counts_.end(), 0);
        for (uint32_t id = blocks_.size(); id > 0; --id) {
            free_ids_.push_back(id - 1);
        }
    }

private:
    uint32_t block_size_ = 0;
    std::vector<llama_kv_block> blocks_;
    std::vector<uint32_t> free_ids_;
    std::vector<uint32_t> ref_counts_;
};

class llama_kv_block_table {
public:
    void init(uint32_t n_seq_max) {
        seq_pages_.clear();
        seq_pages_.resize(n_seq_max);
        seq_n_tokens_.assign(n_seq_max, 0);
        n_entries_ = 0;
    }

    bool insert(llama_seq_id seq_id, uint32_t page, uint32_t block_id) {
        if (!valid_seq(seq_id)) {
            return false;
        }

        auto & pages = seq_pages_[seq_id];
        if (page >= pages.size()) {
            pages.resize(page + 1, LLAMA_KV_BLOCK_ID_NONE);
        }
        if (pages[page] == LLAMA_KV_BLOCK_ID_NONE) {
            ++n_entries_;
        }
        pages[page] = block_id;
        return true;
    }

    uint32_t lookup(llama_seq_id seq_id, uint32_t page) const {
        if (!valid_seq(seq_id)) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }

        const auto & pages = seq_pages_[seq_id];
        return page < pages.size() ? pages[page] : LLAMA_KV_BLOCK_ID_NONE;
    }

    uint32_t erase_page(llama_seq_id seq_id, uint32_t page) {
        if (!valid_seq(seq_id) || page >= seq_pages_[seq_id].size()) {
            return LLAMA_KV_BLOCK_ID_NONE;
        }

        auto & value = seq_pages_[seq_id][page];
        const uint32_t old = value;
        if (old != LLAMA_KV_BLOCK_ID_NONE) {
            value = LLAMA_KV_BLOCK_ID_NONE;
            --n_entries_;
        }
        return old;
    }

    std::vector<uint32_t> erase_seq(llama_seq_id seq_id) {
        if (!valid_seq(seq_id)) {
            return {};
        }

        std::vector<uint32_t> result;
        auto & pages = seq_pages_[seq_id];
        for (uint32_t id : pages) {
            if (id != LLAMA_KV_BLOCK_ID_NONE) {
                result.push_back(id);
                --n_entries_;
            }
        }
        pages.clear();
        seq_n_tokens_[seq_id] = 0;
        return result;
    }

    uint32_t n_tokens(llama_seq_id seq_id) const {
        return valid_seq(seq_id) ? seq_n_tokens_[seq_id] : 0;
    }

    bool set_n_tokens(llama_seq_id seq_id, uint32_t n_tokens) {
        if (!valid_seq(seq_id)) {
            return false;
        }
        seq_n_tokens_[seq_id] = n_tokens;
        return true;
    }

    const std::vector<uint32_t> * get(llama_seq_id seq_id) const {
        return valid_seq(seq_id) ? &seq_pages_[seq_id] : nullptr;
    }

    size_t size() const {
        return n_entries_;
    }

    uint32_t max_mapped_page_plus1() const {
        uint32_t result = 0;
        for (const auto & pages : seq_pages_) {
            for (uint32_t page = pages.size(); page > result; --page) {
                if (pages[page - 1] != LLAMA_KV_BLOCK_ID_NONE) {
                    result = page;
                    break;
                }
            }
        }
        return result;
    }

    void clear() {
        for (auto & pages : seq_pages_) {
            pages.clear();
        }
        std::fill(seq_n_tokens_.begin(), seq_n_tokens_.end(), 0);
        n_entries_ = 0;
    }

    static uint32_t logical_page(llama_pos pos, uint32_t block_size) {
        assert(pos >= 0);
        return uint32_t(pos) / block_size;
    }

    static uint32_t intra_offset(llama_pos pos, uint32_t block_size) {
        assert(pos >= 0);
        return uint32_t(pos) % block_size;
    }

private:
    bool valid_seq(llama_seq_id seq_id) const {
        return seq_id >= 0 && uint32_t(seq_id) < seq_pages_.size();
    }

    std::vector<std::vector<uint32_t>> seq_pages_;
    std::vector<uint32_t> seq_n_tokens_;
    size_t n_entries_ = 0;
};
