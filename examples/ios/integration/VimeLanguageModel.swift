import Foundation
import CoreML
import CryptoKit
import CSentencePiece

/// Confined to the candidate worker's serial queue. No mutable state crosses actors.
nonisolated final class VimeSentenceTokenizer {
    private let processor: UnsafeMutableRawPointer
    init(url: URL) throws {
        guard let p = url.path.withCString({ vime_sp_load($0) }) else { throw VimeLanguageModel.Failure.tokenizer }
        processor = p
    }
    deinit { vime_sp_destroy(processor) }
    func encode(_ text: String) throws -> [Int32] {
        var ids: UnsafeMutablePointer<Int32>?
        var count = 0
        let bytes = Array(text.utf8)
        let ok = bytes.withUnsafeBytes { data in
            vime_sp_encode(processor, data.baseAddress?.assumingMemoryBound(to: CChar.self), data.count, &ids, &count)
        }
        defer { if let ids { vime_sp_free(ids) } }
        guard ok == 1 else { throw VimeLanguageModel.Failure.tokenizer }
        return ids.map { Array(UnsafeBufferPointer(start: $0, count: count)) } ?? []
    }
    func decode(_ ids: [Int32]) throws -> String {
        var text: UnsafeMutablePointer<CChar>?
        var bytes = 0
        let ok = ids.withUnsafeBufferPointer { vime_sp_decode(processor, $0.baseAddress, $0.count, &text, &bytes) }
        defer { if let text { vime_sp_free(text) } }
        guard ok == 1, let text,
              let result = String(bytes: UnsafeRawBufferPointer(start: text, count: bytes), encoding: .utf8)
        else { throw VimeLanguageModel.Failure.tokenizer }
        return result
    }
}

nonisolated final class VimeLanguageModel {
    enum Failure: Error { case resources, integrity, tokenizer, ineligible, tensor, cancelled }
    struct Manifest: Decodable {
        let minimum_ios: Int
        let vocab_size: Int
        let context_length: Int
        let tokenizer_sha256: String
        let compiled_files_sha256: [String: String]
    }
    let tokenizer: VimeSentenceTokenizer
    private let model: MLModel
    static let vocabulary = 16384
    static let contextLength = 128
    init(bundle: Bundle = .main) throws {
        guard let modelURL = bundle.url(forResource: "TinyJapaneseINT8", withExtension: "mlmodelc"),
              let tokenizerURL = bundle.url(forResource: "VimeJapaneseTokenizer", withExtension: "model"),
              let manifestURL = bundle.url(forResource: "VimeLMManifest", withExtension: "json") else { throw Failure.resources }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.minimum_ios == 18, manifest.vocab_size == Self.vocabulary,
              manifest.context_length == Self.contextLength,
              try Self.sha256(tokenizerURL) == manifest.tokenizer_sha256,
              !manifest.compiled_files_sha256.isEmpty else { throw Failure.integrity }
        for (name, digest) in manifest.compiled_files_sha256 {
            guard !name.hasPrefix("/"), !name.split(separator: "/").contains(".."),
                  try Self.sha256(modelURL.appendingPathComponent(name)) == digest else { throw Failure.integrity }
        }
        tokenizer = try VimeSentenceTokenizer(url: tokenizerURL)
        // FP32 compute was evaluated on CPU. Other compute units need their own quality/device checks.
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }
    private static func exactText(_ a: String, _ b: String) -> Bool { a.utf8.elementsEqual(b.utf8) }
    private static func sha256(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    }
    func predict(_ ids: [Int32]) throws -> MLMultiArray {
        guard !ids.isEmpty, ids.count <= Self.contextLength,
              ids.allSatisfy({ $0 >= 0 && $0 < Self.vocabulary }) else { throw Failure.tensor }
        let input = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        let pointer = input.dataPointer.bindMemory(to: Int32.self, capacity: ids.count)
        for (i, id) in ids.enumerated() { pointer[i] = id }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": input]))
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
              logits.dataType == .float32,
              logits.shape.map(\.intValue) == [1, ids.count, Self.vocabulary] else { throw Failure.tensor }
        return logits
    }
    static func logProbabilities(_ logits: MLMultiArray, row: Int) throws -> [Double] {
        guard row >= 0, row < logits.shape[1].intValue else { throw Failure.tensor }
        let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
        let offset = row * logits.strides[1].intValue
        let stride = logits.strides[2].intValue
        var values = (0..<vocabulary).map { Double(pointer[offset + $0 * stride]) }
        guard values.allSatisfy(\.isFinite), let maximum = values.max() else { throw Failure.tensor }
        let normalizer = maximum + log(values.reduce(0) { $0 + exp($1 - maximum) })
        for i in values.indices { values[i] -= normalizer }
        return values
    }
    /// Exactly the frozen scorer: joint encoding, common prefix, full-vocabulary log-softmax,
    /// BOS=2, right PAD=0, summed suffix scores, no candidate-final EOS or truncation.
    func scores(context: String, candidates: [String], cancelled: () -> Bool = { false }) throws -> [Double] {
        guard !candidates.isEmpty, Set(candidates.map { Data($0.utf8) }).count == candidates.count,
              candidates.allSatisfy({ !$0.isEmpty }) else { throw Failure.ineligible }
        let contextTokens = try tokenizer.encode(context)
        guard Self.exactText(try tokenizer.decode(contextTokens), context),
              contextTokens.count + 1 <= Self.contextLength else { throw Failure.ineligible }
        let contextIDs: [Int32] = [2] + contextTokens
        let sequences: [[Int32]] = try candidates.map { text in
            let content = try tokenizer.encode(context + text)
            guard Self.exactText(try tokenizer.decode(content), context + text), !content.isEmpty,
                  content.count <= Self.contextLength else { throw Failure.ineligible }
            return [2] + content
        }
        var common = 0
        for i in 0..<([contextIDs] + sequences).map(\.count).min()! {
            if sequences.allSatisfy({ $0[i] == contextIDs[i] }) { common += 1 } else { break }
        }
        guard common >= 1, sequences.allSatisfy({ $0.count > common }) else { throw Failure.ineligible }
        let length = sequences.map { $0.count - 1 }.max()!
        var sums = [Double]()
        for sequence in sequences {
            if cancelled() { throw Failure.cancelled }
            let input = Array(sequence.dropLast()) + Array(repeating: Int32(0), count: length - sequence.count + 1)
            let sum: Double = try autoreleasepool {
                let logits = try predict(input)
                var sum = 0.0
                for row in (common - 1)..<(sequence.count - 1) {
                    let probabilities = try Self.logProbabilities(logits, row: row)
                    sum += probabilities[Int(sequence[row + 1])]
                }
                return sum
            }
            guard sum.isFinite else { throw Failure.tensor }
            sums.append(sum) // Release each candidate's full logits before the next prediction.
        }
        return sums
    }
    /// Rerank full, literal dictionary conversions in their existing slots. Partial conversion,
    /// script choice, reading alternatives and corrections retain the engine's UI semantics.
    func rerank(_ values: [CandidateSnapshot], context: String, katakana: Bool,
                cancelled: @escaping () -> Bool) -> [CandidateSnapshot] {
        guard !katakana, !cancelled() else { return values }
        let slots = values.indices.filter { values[$0].source == .conversion && values[$0].fullConsumption
            && values[$0].exactReading && values[$0].lexical }
        guard slots.count > 1 else { return values }
        let started = ProcessInfo.processInfo.systemUptime
        let stop = { cancelled() || ProcessInfo.processInfo.systemUptime - started > 0.250 }
        guard let sums = try? scores(context: context, candidates: slots.map { values[$0].text }, cancelled: stop),
              !stop() else { return values }
        let order = sums.indices.sorted { sums[$0] == sums[$1] ? $0 < $1 : sums[$0] > sums[$1] }
        var result = values
        for (i, slot) in slots.enumerated() { result[slot] = values[slots[order[i]]] }
        return result
    }
    struct Suggestion: Sendable {
        let text: String
        let tokenIDs: [Int32]
        let logProbabilitySum: Double
        let incomplete: Bool
    }
    /// Fixed beam search matching the Mac demo. UI insertion always requires a selection.
    /// Kept separate from candidate ranking; automatic keyboard suggestions await quality/device review.
    func suggestions(prompt: String, count: Int = 5, maxTokens: Int = 8,
                     cancelled: () -> Bool = { false }) throws -> [Suggestion] {
        guard !prompt.isEmpty, (1...5).contains(count), (1...8).contains(maxTokens) else { throw Failure.ineligible }
        let content = try tokenizer.encode(prompt)
        guard Self.exactText(try tokenizer.decode(content), prompt), content.count + 1 <= Self.contextLength else { throw Failure.ineligible }
        let prefix: [Int32] = [2] + content
        struct Beam { let ids: [Int32]; let score: Double; var incomplete = false }
        var active = [Beam(ids: [], score: 0)]
        var finished = [Beam]()
        let punctuation = CharacterSet(charactersIn: "。！？!?")
        let trim = CharacterSet(charactersIn: " 、，,。！？!?\n\t")
        func decodedSuffix(_ ids: [Int32]) throws -> String? {
            let text = try tokenizer.decode(content + ids)
            guard text.utf8.starts(with: prompt.utf8), !text.contains("\u{fffd}") else { return nil }
            return String(decoding: text.utf8.dropFirst(prompt.utf8.count), as: UTF8.self)
        }
        for depth in 0..<maxTokens {
            var expanded = [Beam]()
            for beam in active {
                if cancelled() { throw Failure.cancelled }
                if prefix.count + beam.ids.count > Self.contextLength {
                    finished.append(Beam(ids: beam.ids, score: beam.score, incomplete: true)); continue
                }
                var probabilities = try autoreleasepool {
                    let logits = try predict(prefix + beam.ids)
                    return try Self.logProbabilities(logits, row: prefix.count + beam.ids.count - 1)
                }
                for token in [0, 1, 2] { probabilities[token] = -.infinity }
                if (try decodedSuffix(beam.ids) ?? "").trimmingCharacters(in: trim).isEmpty {
                    probabilities[3] = -.infinity
                }
                var top = [Int]()
                for token in probabilities.indices {
                    if top.count == 8, probabilities[token] <= probabilities[top.last!] { continue }
                    top.append(token)
                    top.sort { probabilities[$0] == probabilities[$1] ? $0 < $1 : probabilities[$0] > probabilities[$1] }
                    if top.count > 8 { top.removeLast() }
                }
                for token in top where probabilities[token].isFinite {
                    let ids = beam.ids + [Int32(token)]
                    let next = Beam(ids: ids, score: beam.score + probabilities[token])
                    let suffix = try decodedSuffix(ids)
                    let stops = suffix?.trimmingCharacters(in: .whitespacesAndNewlines).unicodeScalars.last
                        .map { punctuation.contains($0) } ?? false
                    if token == 3 || stops { finished.append(next) } else { expanded.append(next) }
                }
            }
            active = Array(expanded.sorted { $0.score > $1.score }.prefix(8))
            if depth == maxTokens - 1 { finished += active.map { Beam(ids: $0.ids, score: $0.score, incomplete: true) } }
            if active.isEmpty { break }
        }
        if cancelled() { throw Failure.cancelled }
        finished.sort { $0.score / pow(Double(max(1, $0.ids.count)), 0.7) > $1.score / pow(Double(max(1, $1.ids.count)), 0.7) }
        var seen = Set<String>()
        var result = [Suggestion]()
        for beam in finished {
            guard let raw = try decodedSuffix(beam.ids), !raw.trimmingCharacters(in: trim).isEmpty else { continue }
            let key = raw.trimmingCharacters(in: trim)
            guard seen.insert(key).inserted else { continue }
            let end = raw.firstIndex { "。！？!?\n".contains($0) }
            let text = end.map { String(raw[...$0]) } ?? raw
            result.append(Suggestion(text: text, tokenIDs: beam.ids, logProbabilitySum: beam.score,
                                     incomplete: beam.incomplete && !(text.unicodeScalars.last.map { punctuation.contains($0) } ?? false)))
            if result.count == count { break }
        }
        return result
    }

}
