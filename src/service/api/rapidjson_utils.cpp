#include "3rd_party/rapidjson/include/rapidjson/document.h"
#include "3rd_party/rapidjson/include/rapidjson/allocators.h"
#include "3rd_party/rapidjson/include/rapidjson/writer.h"
// #include "3rd_party/rapidjson/include/rapidjson/memorybuffer.h"
#include "rapidjson_utils.h"
#include "data/alignment.h"
#include "service/common/translation_job.h"
#include "service/api/output_options.h"

namespace rapidjson {

Value* ensure_path(Value& node, MemoryPoolAllocator<>& alloc, char const* key){
  if (!node.IsObject())
    return NULL;
  auto m = node.FindMember(key);
  if (m != node.MemberEnd())
    return &(m->value);
  return &(node.AddMember(StringRef(key),{},alloc)[key]);
}

std::string get(const Value& D, const char* key, std::string const& dflt) {
  auto m = D.FindMember(key);
  return m != D.MemberEnd() ? m->value.GetString() : dflt;
}

int get(const Value& D, const char* key, const int& dflt) {
  auto m = D.FindMember(key);
  return m != D.MemberEnd() ? m->value.GetInt() : dflt;
}

bool get(const Value& D, const char* key, const bool& dflt) {
  auto m = D.FindMember(key);
  return m != D.MemberEnd() ? m->value.GetBool() : dflt;
}

Value words2json(const marian::Words& words,
                 const marian::Vocab& V,
                 MemoryPoolAllocator<>& alloc) {
  Value ret(kArrayType);
  for (const auto& w: words) {
    ret.PushBack(Value(V[w].c_str(), alloc),alloc);
  }
  return ret;
}

Value hyp2json(const marian::Result& nbestlist_item,
               const marian::server::TranslationService& service,
               const marian::server::OutputOptions& opts,
               MemoryPoolAllocator<>& alloc) {

  // We assume that word IDs are already in the left-to-right
  // order in the nbestlist_item; see Job::finish()
  const marian::Words& ttok_ids = std::get<0>(nbestlist_item);
  const marian::Vocab& V = *service.vocab(-1);
  std::string translation = V.decode(ttok_ids);
  if (opts.noDetails()) {
    // No details requested, just return the translation
    Value T(translation.c_str(), translation.size(), alloc);
    // @TODO: deal with errors and warnings
    return T;
  }

  // Details requested
  auto hyp = std::get<1>(nbestlist_item);
  Value ret(kObjectType);

  // @TODO: add warnings and errors, if any

  // Sentence score and translation are always included
  ret.AddMember("sentenceScore",std::get<2>(nbestlist_item),alloc);
  ret.AddMember("translation",Value(translation.c_str(),alloc),alloc);

  if (opts.withTokenization) {
    ret.AddMember("translationTokenized",words2json(ttok_ids,V,alloc),alloc);
  }
  if (opts.withWordAlignment || opts.withSoftAlignment) {
    // We currently handle only translation with single input, but
    // in principle Marian allows multiple inputs.
    // We reflect that in the structure and return an array
    // of mappings from target to source positions.
    auto softAlign = hyp->tracebackAlignment();
    if (service.isRight2LeftDecoder())
      std::reverse(softAlign.begin(), softAlign.end());
    if (opts.withWordAlignment) {
      auto WA = marian::data::ConvertSoftAlignToHardAlign(softAlign);
      std::vector<int> alnvec(ttok_ids.size(),-1); // trgPos -> srcPos
      for (auto& p: WA)
        alnvec[p.tgtPos] = p.srcPos;
      Value waNode(kArrayType);
      waNode.PushBack(Value(kArrayType).Move(),alloc); // see long comment above
      waNode[0].Reserve(alnvec.size(),alloc);
      for (int p: alnvec)
        waNode[0].PushBack(p,alloc);
      ret.AddMember("wordAlignment", waNode.Move(), alloc);
    }
    if (opts.withSoftAlignment) {
      Value saNode(kArrayType);
      saNode.PushBack(Value(kArrayType).Move(),alloc); // see long comment above
      saNode[0].Reserve(ttok_ids.size(),alloc);
      for (const auto& s: softAlign) {
        Value row(kArrayType);
        row.Reserve(s.size(),alloc);
        for (const auto& v: s) {
          row.PushBack(v,alloc);
        }
        saNode[0].PushBack(row.Move(),alloc);
      }
      ret.AddMember("softAlignment", saNode.Move(), alloc);
    }
  }

  if (opts.withWordScores) {
    Value ws(kArrayType);
    std::vector<float> wscores = hyp->getScoreBreakdown();
    if (service.isRight2LeftDecoder())
      std::reverse(wscores.begin(), wscores.end());
    ws.Reserve(wscores.size(),alloc);
    for (const auto& v: wscores)
      ws.PushBack(v,alloc);
    ret.AddMember("wordScores",ws.Move(),alloc);
  }
  return ret;
}

Value job2json(const marian::server::Job& job,
               const marian::server::TranslationService& service,
               const marian::server::OutputOptions& opts,
               MemoryPoolAllocator<>& alloc) {

  Value ret(kObjectType);
  const marian::Vocab& V1 = *service.vocab(0);
  if (opts.noDetails() && job.nbestlist_size == 1) {
    const auto& T = job.translation;
    return Value(&T[0],T.size(),alloc);
  }
  else {
    LOG(trace, "NBest list size is {}", job.nbestlist_size);
  }


  if (opts.withOriginal) {
    // Currently, we assume single input per sentence.
    // The TranslationJob class stores input in an array,
    // to accommodate translation with multiple inputs
    // eventually.
    Value a(kArrayType);
    for (const auto& i: job.input) {
      a.PushBack(Value(i.c_str(),alloc), alloc);
    }
    ret.AddMember("original", a.Move(), alloc);
  }

  if (opts.withTokenization) {
    Value oriTok(kArrayType);
    for (const auto& i: job.input) {
      oriTok.PushBack(words2json(V1.encode(i), V1, alloc), alloc);
    }
    ret.AddMember("originalTokenized", oriTok.Move(), alloc);
  }

  Value nbest(kArrayType);
  nbest.Reserve(job.nbestlist_size, alloc);
  for (const auto& i: job.nbest) {
    nbest.PushBack(hyp2json(i, service, opts, alloc), alloc);
  }
  ret.AddMember("nBest", nbest.Move(), alloc);
  return ret;
}

std::string serialize(Document const& D) {
  rapidjson::StringBuffer buffer;
  buffer.Clear();
  Writer<rapidjson::StringBuffer> writer(buffer);
  D.Accept(writer);
  return std::string(buffer.GetString(), buffer.GetSize());
}

bool setOptions(marian::server::OutputOptions& opts, const rapidjson::Value& v) {
  if (!v.IsObject()) {
    return false; // an error occurred; should we throw an exception here?
  }
  opts.withWordAlignment = get(v, "returnWordAlignment", opts.withWordAlignment);
  opts.withSoftAlignment = get(v, "returnSoftAlignment", opts.withSoftAlignment);
  opts.withTokenization  = get(v, "returnTokenization",  opts.withTokenization);
  opts.withSentenceScore = get(v, "returnSentenceScore", opts.withSentenceScore);
  opts.withWordScores    = get(v, "returnWordScores",    opts.withWordScores);
  opts.withTokenization |= opts.withWordAlignment || opts.withSoftAlignment;
  opts.withOriginal = get(v, "returnOriginal", opts.withOriginal);
  return true;
}



} // end of namespace rapidjson
