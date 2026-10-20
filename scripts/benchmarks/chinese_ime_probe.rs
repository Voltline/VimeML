//! Research adapter around the unmodified upstream Apache-2.0 reference crate.
use chinese_ime_lm::{CandidateFacts, Reranker, SentenceModel};
use serde::Deserialize;
use serde_json::json;
use std::{fs, io::{self, BufRead, Write}, sync::Arc, time::Instant};

#[derive(Deserialize)]
struct Candidate { text: String, answers_key: bool, trusted_dictionary_hit: bool }
#[derive(Deserialize)]
struct Case { id: String, context: String, candidates: Vec<Candidate> }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let start = Instant::now();
    let bytes = fs::read(&args[1])?;
    let model = Arc::new(SentenceModel::load(&bytes)?);
    drop(bytes); // Exclude a redundant copy of the file from loaded working set.
    println!("{}", json!({"event":"loaded", "load_ms":start.elapsed().as_secs_f64()*1000.0}));
    io::stdout().flush()?;
    let mut ack = String::new();
    io::stdin().read_line(&mut ack)?; // Parent samples Windows working set here.
    if args.get(3).map(String::as_str) == Some("trace") {
        let cases: Vec<Case> = io::BufReader::new(fs::File::open(&args[2])?).lines()
            .map(|line| Ok(serde_json::from_str::<Case>(&line?)?))
            .collect::<Result<_,Box<dyn std::error::Error>>>()?;
        let mut samples = vec![Vec::<f64>::new(); cases.len()];
        let mut error = 0.0f32;
        // Each repetition starts with a fresh cache and visits every growing prefix once.
        // Repeating only the final candidate list would measure an exact-result cache hit.
        for repetition in 0..=30 {
            let mut reranker = Reranker::new(Arc::clone(&model));
            for (step, case) in cases.iter().enumerate() {
                let texts: Vec<&str> = case.candidates.iter().map(|c| c.text.as_str()).collect();
                let start = Instant::now();
                let scores = reranker.log_probabilities(&case.context, &texts);
                let elapsed = start.elapsed().as_secs_f64()*1000.0;
                if repetition > 0 { samples[step].push(elapsed); }
                else {
                    let fresh = model.score(&case.context, &texts);
                    for (a,b) in scores.iter().zip(&fresh) { error = error.max((a-b).abs()); }
                }
            }
        }
        for (step, values) in samples.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            println!("{}",json!({"step":step+1,"p50_ms":values[14],"p95_ms":values[28],
                "samples_ms":values,"cached_vs_fresh_max_abs_error":error}));
        }
        return Ok(());
    }
    for line in io::BufReader::new(fs::File::open(&args[2])?).lines() {
        let case: Case = serde_json::from_str(&line?)?;
        let texts: Vec<&str> = case.candidates.iter().map(|c| c.text.as_str()).collect();
        let mut reranker = Reranker::new(Arc::clone(&model));
        let start = Instant::now();
        let chosen = reranker.best_where(&case.context, &texts, |i| CandidateFacts {
            answers_key:case.candidates[i].answers_key,
            trusted_dictionary_hit:case.candidates[i].trusted_dictionary_hit,
        }).unwrap_or(0);
        let gated_ms = start.elapsed().as_secs_f64()*1000.0;
        let full: Vec<usize> = case.candidates.iter().enumerate()
            .filter_map(|(i,c)| c.answers_key.then_some(i)).collect();
        let comparable: Vec<&str> = full.iter().map(|&i| texts[i]).collect();
        let scores = model.score(&case.context, &comparable);
        let no_context_scores = model.score("", &comparable);
        let best = |values: &[f32]| -> Option<usize> {
            let mut best = 0;
            if values.is_empty() { return None; }
            for i in 1..values.len() { if values[i] > values[best] { best = i; } }
            Some(full[best])
        };
        println!("{}",json!({"id":case.id,"chosen":chosen,"gated_ms":gated_ms,
            "comparable":full,"scores":scores,"no_context_scores":no_context_scores,
            "ungated":best(&scores),"no_context":best(&no_context_scores)}));
    }
    Ok(())
}
