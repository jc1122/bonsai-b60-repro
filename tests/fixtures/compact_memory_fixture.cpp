#include "llama-memory-recurrent.h"
#include "llama-io.h"
#include "llama-vocab.h"
#include "model_cpu_adapter.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static size_t checks = 0;
static void require(bool value, const char * label) {
    ++checks;
    if (!value) {
        throw std::runtime_error(label);
    }
}

void llama_log_internal(ggml_log_level, const char *, ...) {}

const std::string & llama_vocab::token_to_piece(llama_token) const {
    throw std::logic_error("tokenizer adapter must not run");
}

static const char * reg_name(ggml_backend_reg_t) { return "SYCL"; }
static const char * dev_name(ggml_backend_dev_t) { return "TEST-CPU-BUFFER"; }
static ggml_backend_buffer_type_t cpu_buft(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }
static ggml_backend_reg test_reg = { GGML_BACKEND_API_VERSION, {}, nullptr };
static ggml_backend_device test_dev = { {}, &test_reg, nullptr };

static llama_model make_model() {
    llama_model model;
    model.device = &test_dev;
    model.hparams.n_layer_all = 2;
    model.hparams.ssm_d_conv = 4;
    model.hparams.ssm_d_state = 128;
    model.hparams.ssm_dt_rank = 1;
    model.hparams.ssm_d_inner = 128;
    model.hparams.ssm_n_group = 1;
    return model;
}

static void set_compact(bool enabled) {
    setenv("LLAMA_GDN_REPLAY", "1", 1);
    setenv("LLAMA_GDN_REPLAY_MULTI", "1", 1);
    setenv("LLAMA_GDN_REPLAY_COMPACT", enabled ? "1" : "0", 1);
    unsetenv("GGML_SYCL_GDN_CHUNKED");
}

static std::unique_ptr<llama_memory_recurrent> cache(llama_model & model, int slots, int depth, bool compact) {
    set_compact(compact);
    return std::make_unique<llama_memory_recurrent>(model, GGML_TYPE_F32, GGML_TYPE_F32, true, slots, slots, depth, nullptr);
}

static ggml_tensor * record_tensor(llama_memory_recurrent & mem, int layer) {
#if HAS_COMPACT
    return mem.rpl_compact ? mem.rpl_records_l[layer] : mem.s_l[layer];
#else
    return mem.s_l[layer];
#endif
}

static size_t record_offset(llama_memory_recurrent & mem, int layer, int cell, int bank) {
#if HAS_COMPACT
    if (mem.rpl_compact) {
        return cell * mem.rpl_records_l[layer]->nb[1] + bank * mem.n_rs_seq * mem.rpl_R * sizeof(float);
    }
#endif
    return ((mem.size + cell) * mem.s_l[layer]->ne[0] + bank * mem.n_rs_seq * mem.rpl_R) * sizeof(float);
}

static std::vector<uint32_t> get(ggml_tensor * tensor, size_t offset, size_t words) {
    require(offset + words * 4 <= ggml_nbytes(tensor), "tensor get bounds");
    std::vector<uint32_t> data(words);
    ggml_backend_tensor_get(tensor, data.data(), offset, words * 4);
    return data;
}

static void put(ggml_tensor * tensor, size_t offset, const std::vector<uint32_t> & words) {
    require(offset + words.size() * 4 <= ggml_nbytes(tensor), "tensor put bounds");
    ggml_backend_tensor_set(tensor, words.data(), offset, words.size() * 4);
}

static std::vector<uint32_t> pattern(size_t words, uint32_t seed) {
    std::vector<uint32_t> out(words);
    for (size_t i = 0; i < words; ++i) {
        out[i] = 0x3e000000u + ((seed * 65537u + i * 17u) & 0x007fffffu);
    }
    return out;
}

static void fill(llama_memory_recurrent & mem, int pending, int bank) {
    mem.head = 0;
    mem.n = mem.size;
    mem.used = mem.size;
    for (uint32_t c = 0; c < mem.size; ++c) {
        auto & cell = mem.cells[c];
        cell.pos = 100 + c;
        cell.src = c;
        cell.src0 = c;
        cell.tail = c;
        cell.seq_id = { static_cast<llama_seq_id>(c) };
        mem.rpl_n[c] = pending;
        mem.rpl_bank[c] = bank;
        mem.rs_idx[c] = mem.n_rs_seq - pending;
    }
    for (size_t layer = 0; layer < mem.s_l.size(); ++layer) {
        auto * s = mem.s_l[layer];
        auto * r = mem.r_l[layer];
        put(s, 0, pattern(ggml_nbytes(s) / 4, 999 + layer));
        put(r, 0, pattern(ggml_nbytes(r) / 4, 777 + layer));
        for (uint32_t c = 0; c < mem.size; ++c) {
            put(s, c * s->nb[1], pattern(s->ne[0], 10 + c + layer * 3));
            for (int b = 0; b < 2; ++b) {
                put(record_tensor(mem, layer), record_offset(mem, layer, c, b),
                        pattern(mem.n_rs_seq * mem.rpl_R, 100 + c * 2 + b + layer * 7));
            }
        }
    }
}

struct writer : llama_io_write_i {
    std::vector<uint8_t> data;
    void write(const void * src, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(src);
        data.insert(data.end(), p, p + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        require(offset + size <= ggml_nbytes(tensor), "serialize write bounds");
        size_t start = data.size();
        data.resize(start + size);
        ggml_backend_tensor_get(tensor, data.data() + start, offset, size);
    }
    size_t n_bytes() override { return data.size(); }
};

struct reader : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t cursor = 0;
    explicit reader(const std::vector<uint8_t> & data) : data(data) {}
    void read(void * dst, size_t size) override {
        require(cursor + size <= data.size(), "serialize input bounds");
        std::memcpy(dst, data.data() + cursor, size);
        cursor += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        require(cursor + size <= data.size(), "serialize tensor input bounds");
        require(offset + size <= ggml_nbytes(tensor), "serialize read bounds");
        ggml_backend_tensor_set(tensor, data.data() + cursor, offset, size);
        cursor += size;
    }
    size_t n_bytes() override { return cursor; }
};

static std::vector<uint8_t> save(llama_memory_recurrent & mem, int seq = -1) {
    writer output;
    mem.state_write(output, seq);
    return output.data;
}

static void load(llama_memory_recurrent & mem, const std::vector<uint8_t> & bytes, int seq = -1) {
    reader input(bytes);
    mem.state_read(input, seq);
    require(input.cursor == bytes.size(), "wire consumed exactly");
}

static void equal_metadata(const llama_memory_recurrent & a, const llama_memory_recurrent & b) {
    require(a.head == b.head && a.used == b.used && a.n == b.n, "memory metadata");
    require(a.rs_idx == b.rs_idx && a.rpl_n == b.rpl_n && a.rpl_bank == b.rpl_bank, "replay metadata");
    for (uint32_t i = 0; i < a.size; ++i) {
        require(a.cells[i].pos == b.cells[i].pos && a.cells[i].src == b.cells[i].src &&
                a.cells[i].src0 == b.cells[i].src0 && a.cells[i].tail == b.cells[i].tail &&
                a.cells[i].seq_id == b.cells[i].seq_id, "cell metadata");
    }
}

static void wire_cases(llama_model & model) {
    for (int slots : { 1, 2, 3 }) {
        for (int depth : { 2, 3, 4 }) {
            for (int pending = 0; pending <= depth; ++pending) {
                for (int bank : { 0, 1 }) {
                    auto padded = cache(model, slots, depth, false);
                    auto compact = cache(model, slots, depth, true);
                    require(padded->rpl && compact->rpl, "replay enabled in CPU adapter");
                    require(compact->s_l[0]->ne[1] == slots * (slots == 3 ? 1 : 2), "compact allocation active");
#if HAS_COMPACT
                    require(compact->rpl_compact == (slots == 3), "exact three-cell compact guard");
                    require((compact->rpl_records_l[0] != nullptr) == (slots == 3), "unsupported cells keep padded banks");
#endif
                    require(padded->s_l[0]->ne[1] == 2 * slots, "off allocation unchanged");
                    fill(*padded, pending, bank);
                    fill(*compact, pending, bank);
                    auto bytes = save(*padded);
                    require(bytes == save(*compact), "padded and compact wire bytes");
                    auto restored_padded = cache(model, slots, depth, false);
                    auto restored_compact = cache(model, slots, depth, true);
                    load(*restored_padded, save(*compact));
                    load(*restored_compact, bytes);
                    require(save(*restored_padded) == save(*restored_compact), "cross-layout restored wire");
                    require(save(*restored_padded) == bytes, "full wire stable after restore");
                    for (int seq = 0; seq < slots; ++seq) {
                        auto one = save(*padded, seq);
                        require(one == save(*compact, seq), "single sequence wire bytes");
                        auto dst_p = cache(model, slots, depth, false);
                        auto dst_c = cache(model, slots, depth, true);
                        load(*dst_p, one, seq);
                        load(*dst_c, one, seq);
                        require(save(*dst_p, seq) == save(*dst_c, seq), "single cross-layout restore");
                    }
                    if (slots == 3) {
                        padded->seq_rm(2, -1, -1);
                        compact->seq_rm(2, -1, -1);
                        padded->seq_cp(0, 2, -1, -1);
                        compact->seq_cp(0, 2, -1, -1);
                        equal_metadata(*padded, *compact);
                        require(save(*padded) == save(*compact), "shared alias wire");
                        auto prefix_p = cache(model, slots, depth, false);
                        auto prefix_c = cache(model, slots, depth, true);
                        load(*prefix_p, save(*padded));
                        load(*prefix_c, save(*compact));
                        auto surviving = save(*prefix_p, 2);
                        auto prefix = save(*prefix_p, 0);
                        load(*prefix_p, prefix, 0);
                        load(*prefix_c, prefix, 0);
                        require(save(*prefix_p, 2) == surviving && save(*prefix_c, 2) == surviving, "prefix restore preserves surviving alias");
                        require(save(*prefix_p) == save(*prefix_c), "prefix restore full wire");
                        if (pending > 0 && pending == depth) {
                            auto p0 = padded->cells[padded->cells[2].tail].pos;
                            require(padded->seq_rm(2, p0, -1) == compact->seq_rm(2, p0, -1), "shared detach result");
                            equal_metadata(*padded, *compact);
                            require(save(*padded) == save(*compact), "detached rollback wire");
                        }
                        padded->seq_keep(0);
                        compact->seq_keep(0);
                        equal_metadata(*padded, *compact);
                        require(save(*padded) == save(*compact), "seq_keep wire");
                    }
                    compact->clear(true);
                    require(compact->used == 0, "clear metadata");
                    for (size_t layer = 0; layer < compact->s_l.size(); ++layer) {
                        auto * records = record_tensor(*compact, layer);
                        auto words = get(records, 0, ggml_nbytes(records) / 4);
                        require(std::all_of(words.begin(), words.end(), [](uint32_t x) { return x == 0; }), "clear record banks");
                    }
                }
            }
        }
    }
}

// The runner inserts the literal production build_rs body here, changing only its signature.
#include "production_build_rs.inc"

static void transport_cases(ggml_backend_t cpu) {
    for (int depth : { 2, 3, 4 }) {
        const int width = 2 * depth * 257;
        for (int active : { 1, 2, 3 }) {
            std::vector<int32_t> perm = { 0, 1, 2 };
            do {
                ggml_init_params params = { 8 * 1024 * 1024, nullptr, false };
                ggml_context * ctx0 = ggml_init(params);
                require(ctx0 != nullptr, "transport arena");
                ggml_cgraph * gf = ggml_new_graph(ctx0);
                auto * records = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, width, 3);
                auto * main = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, active);
                auto * extra = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 3 - active);
                auto original = pattern(width * 3, 555);
                std::memcpy(records->data, original.data(), original.size() * 4);
                std::memcpy(main->data, perm.data(), active * 4);
                if (active < 3) {
                    std::memcpy(extra->data, perm.data() + active, (3 - active) * 4);
                }
                auto * gathered = fixture_build_rs(ctx0, gf, records, main, extra, width, active, 3, 0, 3, -1, ggml_get_rows);
                require(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "transport graph compute");
                for (int row = 0; row < active; ++row) {
                    require(std::memcmp(static_cast<char *>(gathered->data) + row * width * 4,
                                original.data() + perm[row] * width, width * 4) == 0, "gather before cyclic relocation");
                }
                for (int row = active; row < 3; ++row) {
                    require(std::memcmp(static_cast<char *>(records->data) + row * width * 4,
                                original.data() + perm[row] * width, width * 4) == 0, "inactive both-bank relocation");
                }
                for (int bank = 0; bank < 2; ++bank) {
                    std::vector<uint32_t> before(width * 3);
                    std::memcpy(before.data(), records->data, before.size() * 4);
                    const int bank_width = width / 2;
                    auto * fresh = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, bank_width, 1, active);
                    auto fresh_data = pattern(bank_width * active, 888 + bank);
                    std::memcpy(fresh->data, fresh_data.data(), fresh_data.size() * 4);
                    auto * indices = ggml_new_tensor_2d(ctx0, GGML_TYPE_I64, 1, active);
                    for (int row = 0; row < active; ++row) {
                        static_cast<int64_t *>(indices->data)[row] = (bank + row) % 2;
                    }
                    auto * dst = ggml_view_3d(ctx0, records, bank_width, 2, active, bank_width * 4, width * 4, 0);
                    auto * scatter = ggml_set_rows(ctx0, dst, fresh, indices);
                    auto * scatter_graph = ggml_new_graph(ctx0);
                    ggml_build_forward_expand(scatter_graph, scatter);
                    require(ggml_backend_graph_compute(cpu, scatter_graph) == GGML_STATUS_SUCCESS, "bank scatter graph");
                    for (int row = 0; row < active; ++row) {
                        int selected = (bank + row) % 2;
                        require(std::memcmp(static_cast<char *>(records->data) + (row * width + selected * bank_width) * 4,
                                    fresh_data.data() + row * bank_width, bank_width * 4) == 0, "scatter selects own sequence bank");
                        require(std::memcmp(static_cast<char *>(records->data) + (row * width + (selected ^ 1) * bank_width) * 4,
                                    before.data() + row * width + (selected ^ 1) * bank_width, bank_width * 4) == 0, "scatter preserves read bank");
                    }
                    for (int row = active; row < 3; ++row) {
                        require(std::memcmp(static_cast<char *>(records->data) + row * width * 4,
                                    before.data() + row * width, width * 4) == 0, "scatter preserves inactive cells");
                    }
                }
                ggml_free(ctx0);
            } while (std::next_permutation(perm.begin(), perm.end()));
        }
    }
}

static void bookkeeping_cases(llama_model & model) {
    for (int depth : { 2, 3, 4 }) {
        for (int active : { 1, 2, 3 }) {
            for (int tokens : { 1, 2, 5, 16 }) {
                auto padded = cache(model, 3, depth, false);
                auto compact = cache(model, 3, depth, true);
                fill(*padded, depth, 1);
                fill(*compact, depth, 1);
                llama_batch_allocr balloc(1);
                auto ubatch = balloc.ubatch_reserve(tokens, active);
                ubatch.data->seq_id_data.resize(active);
                for (int row = 0; row < active; ++row) {
                    int seq = (row + 2) % 3;
                    ubatch.data->seq_id_data[row] = seq;
                    for (int t = 0; t < tokens; ++t) {
                        int i = row * tokens + t;
                        ubatch.pos[i] = 101 + seq + t;
                        ubatch.n_seq_id[i] = 1;
                        ubatch.seq_id[i] = &ubatch.data->seq_id_data[row];
                    }
                }
                auto before = save(*padded);
                require(padded->prepare({ubatch}) == compact->prepare({ubatch}), "prepare result");
                equal_metadata(*padded, *compact);
                require(save(*padded) == before && save(*compact) == before, "prepare preserves wire state");
                require(padded->find_slot(ubatch) == compact->find_slot(ubatch), "find_slot result");
                equal_metadata(*padded, *compact);
                llama_memory_recurrent_context pctx(padded.get(), {ubatch});
                llama_memory_recurrent_context cctx(compact.get(), {ubatch});
                for (uint32_t row = 0; row < pctx.get_n_rs(); ++row) {
                    require(pctx.rpl_row(row) == cctx.rpl_row(row), "record source cell map");
                    require(pctx.s_copy(row) == cctx.s_copy(row), "convolution rollback source map");
                }
                for (int row = 0; row < active; ++row) {
                    int32_t pn, pb, cn, cb;
                    pctx.rpl_take(row, tokens, pn, pb);
                    cctx.rpl_take(row, tokens, cn, cb);
                    require(pn == cn && pb == cb, "record count and bank consumption");
                }
                equal_metadata(*padded, *compact);
                require(padded->seq_rm(0, -1, -1) == compact->seq_rm(0, -1, -1), "cancel bookkeeping");
                equal_metadata(*padded, *compact);
            }
        }
    }
}

static void guards(llama_model & model) {
    for (bool compact : { false, true }) {
        set_compact(compact);
        llama_memory_recurrent zero(model, GGML_TYPE_F32, GGML_TYPE_F32, true, 3, 3, 0, nullptr);
        require(!zero.rpl && zero.s_l[0]->ne[1] == 3, "depth zero legacy layout");
        llama_memory_recurrent host(model, GGML_TYPE_F32, GGML_TYPE_F32, false, 3, 3, 4, nullptr);
        require(!host.rpl && host.s_l[0]->ne[1] == 15, "host-only legacy layout");
    }
    unsetenv("LLAMA_GDN_REPLAY_COMPACT");
    llama_memory_recurrent missing(model, GGML_TYPE_F32, GGML_TYPE_F32, true, 3, 3, 4, nullptr);
    require(missing.rpl && missing.s_l[0]->ne[1] == 6, "missing env stays padded");
#if HAS_COMPACT
    auto two = cache(model, 2, 4, true);
    require(!two->rpl_compact && two->rpl_records_l[0] == nullptr, "two-cell compact request declines");
    require(two->s_l[0]->ne[1] == 4 && two->s_l[0]->ne[1] != 2,
            "two-cell one-active gather-skip ambiguity is unreachable");
    auto one = cache(model, 1, 4, true);
    require(!one->rpl_compact && one->s_l[0]->ne[1] == 2, "one-cell legacy gather shape retained");
#endif
}

int main() {
    try {
        test_reg.iface.get_name = reg_name;
        test_dev.iface.get_name = dev_name;
        test_dev.iface.get_buffer_type = cpu_buft;
        auto model = make_model();
        wire_cases(model);
        guards(model);
        bookkeeping_cases(model);
        auto cpu = ggml_backend_cpu_init();
        require(cpu != nullptr, "CPU backend");
        transport_cases(cpu);
        ggml_backend_free(cpu);
        std::printf("{\"checks\":%zu,\"passed\":true,\"adapter\":\"CPU-buffer model device; no GPU execution\",\"raw_model_qualified\":false}\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "checks=%zu error=%s\n", checks, error.what());
        return 1;
    }
}
