// Research-only portable librime recorder. Uses the public C API, no system IME.
#define RIME_IMPORTS
#include <rime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) return 2; // shared dir, user dir, schema id, input TSV
    auto* api = rime_get_api();
    RIME_STRUCT(RimeTraits, traits);
    traits.shared_data_dir = argv[1];
    traits.user_data_dir = argv[2];
    traits.app_name = "rime.vimeml_research";
    traits.min_log_level = 2;
    traits.log_dir = argv[2];
    api->setup(&traits);
    api->initialize(&traits);
    std::cerr << "librime " << api->get_version()
              << " octagram=" << bool(api->find_module("octagram")) << '\n';
    if (api->start_maintenance(true)) api->join_maintenance_thread();
    auto session = [&]() {
        auto id = api->create_session();
        if (!api->select_schema(id, argv[3])) { std::cerr << "schema failed\n"; std::exit(3); }
        return id;
    };
    std::ifstream input(argv[4]);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        auto ident = line.substr(0, tab), keys = line.substr(tab + 1);
        auto id = session();
        api->simulate_key_sequence(id, keys.c_str());
        std::vector<std::string> texts;
        RimeCandidateListIterator iterator{};
        if (api->candidate_list_begin(id, &iterator)) {
            while (texts.size() < 9 && api->candidate_list_next(&iterator))
                texts.emplace_back(iterator.candidate.text);
            api->candidate_list_end(&iterator);
        }
        api->destroy_session(id);
        for (size_t i = 0; i < texts.size(); ++i) {
            // A full-key answer commits on selection; a prefix leaves composition.
            id = session();
            api->simulate_key_sequence(id, keys.c_str());
            if (!api->select_candidate(id, i)) return 4;
            RIME_STRUCT(RimeCommit, commit);
            bool complete = false;
            if (api->get_commit(id, &commit)) {
                complete = commit.text && texts[i] == commit.text;
                api->free_commit(&commit);
            }
            std::cout << ident << '\t' << i << '\t' << texts[i]
                      << '\t' << complete << '\n';
            api->destroy_session(id);
        }
    }
    api->finalize();
}
