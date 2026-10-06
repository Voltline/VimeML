#include "include/VimeSentencePiece.h"
#include "sentencepiece_processor.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
extern "C" void *vime_sp_load(const char *path) {
  try {
    auto p = std::make_unique<sentencepiece::SentencePieceProcessor>();
    if (!p->Load(path).ok() || p->GetPieceSize() != 16384 || p->pad_id() != 0 ||
        p->unk_id() != 1 || p->bos_id() != 2 || p->eos_id() != 3) return nullptr;
    return p.release();
  } catch (...) { return nullptr; }
}
extern "C" void vime_sp_destroy(void *p) { delete static_cast<sentencepiece::SentencePieceProcessor *>(p); }
extern "C" void vime_sp_free(void *p) { std::free(p); }
extern "C" int vime_sp_encode(void *p, const char *text, size_t bytes, int32_t **ids, size_t *count) {
  *ids = nullptr; *count = 0;
  try {
    std::vector<int> result;
    if (!p || !static_cast<sentencepiece::SentencePieceProcessor *>(p)->Encode(std::string(text ? text : "", bytes), &result).ok()) return 0;
    if (!result.empty()) {
      *ids = static_cast<int32_t *>(std::malloc(result.size() * sizeof(int32_t)));
      if (!*ids) return 0;
      for (size_t i = 0; i < result.size(); ++i) (*ids)[i] = result[i];
    }
    *count = result.size(); return 1;
  } catch (...) { return 0; }
}
extern "C" int vime_sp_decode(void *p, const int32_t *ids, size_t count, char **text, size_t *bytes) {
  *text = nullptr; *bytes = 0;
  try {
    std::vector<int> input;
    if (count) input.assign(ids, ids + count);
    std::string result;
    if (!p || !static_cast<sentencepiece::SentencePieceProcessor *>(p)->Decode(input, &result).ok()) return 0;
    *text = static_cast<char *>(std::malloc(result.size() + 1));
    if (!*text) return 0;
    std::memcpy(*text, result.data(), result.size()); (*text)[result.size()] = 0;
    *bytes = result.size(); return 1;
  } catch (...) { return 0; }
}
