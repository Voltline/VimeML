import XCTest
import CoreML
import KanaKanjiConverterModuleWithDefaultDictionary

@MainActor
final class VimeLanguageModelTests: XCTestCase {
    private struct Fixtures: Decodable {
        struct Token: Decodable { let text: String; let ids: [Int32]; let decoded: String }
        struct Score: Decodable { let context: String; let candidates: [String]; let sums: [Double] }
        let tokenization: [Token]
        struct Phrase: Decodable { let prompt: String; let firstText: String; let firstIDs: [Int32]; let firstSum: Double }
        let phrases: [Phrase]
        let scoring: [Score]
    }
    private func fixtures() throws -> Fixtures {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "VimeLMFixtures", withExtension: "json"))
        return try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
    }
    func testNativeSentencePieceMatchesFrozenCorpusAndUnicode() throws {
        let lm = try VimeLanguageModel()
        let data = try fixtures()
        XCTAssertGreaterThan(data.tokenization.count, 1000)
        for item in data.tokenization {
            XCTAssertEqual(try lm.tokenizer.encode(item.text), item.ids, item.text)
            XCTAssertEqual(try lm.tokenizer.decode(item.ids), item.decoded, item.text)
        }
    }
    func testJointSuffixScoresMatchMacINT8IncludingBoundaryRetokenization() throws {
        let lm = try VimeLanguageModel()
        for item in try fixtures().scoring {
            let actual = try lm.scores(context: item.context, candidates: item.candidates)
            XCTAssertEqual(actual.count, item.sums.count)
            for (value, reference) in zip(actual, item.sums) { XCTAssertEqual(value, reference, accuracy: 0.002) }
        }
    }
    func testWholePoolFallbackAndCancellation() throws {
        let lm = try VimeLanguageModel()
        XCTAssertThrowsError(try lm.scores(context: "", candidates: ["東京", "東京"]))
        XCTAssertThrowsError(try lm.scores(context: "", candidates: ["", "東京"]))
        XCTAssertThrowsError(try lm.scores(context: "▁", candidates: ["東京", "大阪"]))
        XCTAssertThrowsError(try lm.scores(context: String(repeating: "東京", count: 200), candidates: ["駅", "都"]))
        XCTAssertThrowsError(try lm.scores(context: "", candidates: ["東京", "大阪"], cancelled: { true }))
        XCTAssertThrowsError(try lm.predict([]))
        XCTAssertThrowsError(try lm.predict([16384]))
        XCTAssertThrowsError(try lm.predict(Array(repeating: 2, count: 129)))
    }
    func testPaddingAndCausality() throws {
        let lm = try VimeLanguageModel()
        let ids: [Int32] = [2] + (try lm.tokenizer.encode("明日の会議までに、"))
        let plain = try lm.predict(ids)
        let padded = try lm.predict(ids + Array(repeating: 0, count: 128 - ids.count))
        let future = try lm.predict(ids + Array(repeating: 4675, count: 128 - ids.count))
        for row in ids.indices {
            let a = try VimeLanguageModel.logProbabilities(plain, row: row)
            let b = try VimeLanguageModel.logProbabilities(padded, row: row)
            let c = try VimeLanguageModel.logProbabilities(future, row: row)
            XCTAssertLessThan(zip(a,b).map { abs($0-$1) }.max()!, 0.0002)
            XCTAssertLessThan(zip(b,c).map { abs($0-$1) }.max()!, 0.0002)
        }
    }
    func testRerankerRetainsMetadataAndExplicitScriptAndCancelledOrders() throws {
        let lm = try VimeLanguageModel()
        let engine = JapaneseCandidateEngine()
        var query = ComposingText()
        RomajiConverter.insert("nihongo", into: &query)
        let values = engine.candidates(for: query, revision: 73, includeCorrections: false)
        XCTAssertFalse(values.isEmpty)
        XCTAssertEqual(lm.rerank(values, context: "", katakana: true, cancelled: { false }).map(\.presentation), values.map(\.presentation))
        XCTAssertEqual(lm.rerank(values, context: "", katakana: false, cancelled: { true }).map(\.presentation), values.map(\.presentation))
        let reordered = lm.rerank(values, context: "", katakana: false, cancelled: { false })
        XCTAssertEqual(Set(reordered.map(\.text)), Set(values.map(\.text)))
        XCTAssertTrue(reordered.allSatisfy { $0.revision == 73 })
        for index in values.indices where !(values[index].source == .conversion && values[index].fullConsumption && values[index].exactReading && values[index].lexical) {
            XCTAssertEqual(reordered[index].presentation, values[index].presentation)
        }
    }
    func testNativeBeamMatchesTwentyMacINT8Prefixes() throws {
        let lm = try VimeLanguageModel()
        for phrase in try fixtures().phrases {
            let suggestions = try lm.suggestions(prompt: phrase.prompt)
            let first = try XCTUnwrap(suggestions.first, phrase.prompt)
            XCTAssertEqual(first.tokenIDs, phrase.firstIDs, phrase.prompt)
            XCTAssertEqual(first.text, phrase.firstText, phrase.prompt)
            XCTAssertEqual(first.logProbabilitySum, phrase.firstSum, accuracy: 0.005, phrase.prompt)
            XCTAssertEqual(Set(suggestions.map(\.text)).count, suggestions.count)
        }
        XCTAssertThrowsError(try lm.suggestions(prompt: "東京", cancelled: { true }))
    }

}
