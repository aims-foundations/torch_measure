# Benchmark Gallery

Heatmap visualizations for all 45 benchmarks (200 response matrices total).

Each image shows the full response matrix: rows are models (sorted by mean score), columns are items (sorted by difficulty), colored by score (red=incorrect, green=correct). Matrices larger than 1000×2000 are downsampled for display.

To regenerate, run `python data/<benchmark>/build.py` which produces the heatmap as part of the build pipeline.

---

## aegis

### aegis
![aegis](aegis/figures/aegis_heatmap.png)

### aegis — unsafe_only
![aegis_unsafe_only](aegis/figures/aegis_unsafe_only_heatmap.png)

## afrimedqa

### afrimedqa — coverage
![coverage](afrimedqa/figures/coverage_heatmap.png)

### afrimedqa — response
![response](afrimedqa/figures/response_heatmap.png)

## agentbench

### agentbench
![agentbench](agentbench/figures/agentbench_heatmap.png)

### agentbench — with_overall
![agentbench_with_overall](agentbench/figures/agentbench_with_overall_heatmap.png)

## agentdojo

### agentdojo
![agentdojo](agentdojo/figures/agentdojo_heatmap.png)

### agentdojo — security
![agentdojo_security](agentdojo/figures/agentdojo_security_heatmap.png)

### agentdojo — utility_under_attack
![agentdojo_utility_under_attack](agentdojo/figures/agentdojo_utility_under_attack_heatmap.png)

## alpacaeval

### alpacaeval
![alpacaeval](alpacaeval/figures/alpacaeval_heatmap.png)

### alpacaeval — preference
![alpacaeval_preference](alpacaeval/figures/alpacaeval_preference_heatmap.png)

## androidworld

### androidworld
![androidworld](androidworld/figures/androidworld_heatmap.png)

## bbq

### bbq
![bbq](bbq/figures/bbq_heatmap.png)

## bfcl

### bfcl
![bfcl](bfcl/figures/bfcl_heatmap.png)

## bigcodebench

### bigcodebench — hard_complete
![bigcodebench_hard_complete](bigcodebench/figures/bigcodebench_hard_complete_heatmap.png)

### bigcodebench — hard_instruct
![bigcodebench_hard_instruct](bigcodebench/figures/bigcodebench_hard_instruct_heatmap.png)

### bigcodebench
![bigcodebench](bigcodebench/figures/bigcodebench_heatmap.png)

### bigcodebench — instruct
![bigcodebench_instruct](bigcodebench/figures/bigcodebench_instruct_heatmap.png)

## bridging_gap

### bridging_gap — model_language
![model_language](bridging_gap/figures/model_language_heatmap.png)

## browsergym

### browsergym
![browsergym](browsergym/figures/browsergym_heatmap.png)

### browsergym — with_stderr
![browsergym_with_stderr](browsergym/figures/browsergym_with_stderr_heatmap.png)

## collab_cxr

### collab_cxr — image_ai
![collab_cxr_image_ai](collab_cxr/figures/collab_cxr_image_ai_heatmap.png)

### collab_cxr — image_ai_history
![collab_cxr_image_ai_history](collab_cxr/figures/collab_cxr_image_ai_history_heatmap.png)

### collab_cxr — image_history
![collab_cxr_image_history](collab_cxr/figures/collab_cxr_image_history_heatmap.png)

### collab_cxr — image_only
![collab_cxr_image_only](collab_cxr/figures/collab_cxr_image_only_heatmap.png)

## cruxeval

### cruxeval — binary
![cruxeval_binary](cruxeval/figures/cruxeval_binary_heatmap.png)

### cruxeval
![cruxeval](cruxeval/figures/cruxeval_heatmap.png)

### cruxeval — input_binary
![cruxeval_input_binary](cruxeval/figures/cruxeval_input_binary_heatmap.png)

### cruxeval — input
![cruxeval_input](cruxeval/figures/cruxeval_input_heatmap.png)

### cruxeval — output_binary
![cruxeval_output_binary](cruxeval/figures/cruxeval_output_binary_heatmap.png)

### cruxeval — output
![cruxeval_output](cruxeval/figures/cruxeval_output_heatmap.png)

## culturaleval

### culturaleval — language_dataset
![language_dataset](culturaleval/figures/language_dataset_heatmap.png)

### culturaleval — region_dataset
![region_dataset](culturaleval/figures/region_dataset_heatmap.png)

## dpai

### dpai — binary_pass50
![dpai_binary_pass50](dpai/figures/dpai_binary_pass50_heatmap.png)

### dpai — blind_score
![dpai_blind_score](dpai/figures/dpai_blind_score_heatmap.png)

### dpai — informed_score
![dpai_informed_score](dpai/figures/dpai_informed_score_heatmap.png)

### dpai — total_score
![dpai_total_score](dpai/figures/dpai_total_score_heatmap.png)

## editbench

### editbench — binary
![editbench_binary](editbench/figures/editbench_binary_heatmap.png)

### editbench
![editbench](editbench/figures/editbench_heatmap.png)

## evalplus

### evalplus
![evalplus](evalplus/figures/evalplus_heatmap.png)

### evalplus — humaneval_base
![evalplus_humaneval_base](evalplus/figures/evalplus_humaneval_base_heatmap.png)

### evalplus — humaneval_plus
![evalplus_humaneval_plus](evalplus/figures/evalplus_humaneval_plus_heatmap.png)

### evalplus — mbpp_base
![evalplus_mbpp_base](evalplus/figures/evalplus_mbpp_base_heatmap.png)

### evalplus — mbpp_plus
![evalplus_mbpp_plus](evalplus/figures/evalplus_mbpp_plus_heatmap.png)

## gaia

### gaia — hal_continuous
![gaia_hal_continuous](gaia/figures/gaia_hal_continuous_heatmap.png)

### gaia — hal
![gaia_hal](gaia/figures/gaia_hal_heatmap.png)

## genai_learning

### genai_learning — exam_augmented
![genai_learning_exam_augmented](genai_learning/figures/genai_learning_exam_augmented_heatmap.png)

### genai_learning — exam_control
![genai_learning_exam_control](genai_learning/figures/genai_learning_exam_control_heatmap.png)

### genai_learning — exam_vanilla
![genai_learning_exam_vanilla](genai_learning/figures/genai_learning_exam_vanilla_heatmap.png)

### genai_learning — practice_augmented
![genai_learning_practice_augmented](genai_learning/figures/genai_learning_practice_augmented_heatmap.png)

### genai_learning — practice_control
![genai_learning_practice_control](genai_learning/figures/genai_learning_practice_control_heatmap.png)

### genai_learning — practice_vanilla
![genai_learning_practice_vanilla](genai_learning/figures/genai_learning_practice_vanilla_heatmap.png)

## haiid

### haiid — art_post_ai
![haiid_art_post_ai](haiid/figures/haiid_art_post_ai_heatmap.png)

### haiid — art_post_human
![haiid_art_post_human](haiid/figures/haiid_art_post_human_heatmap.png)

### haiid — art_pre
![haiid_art_pre](haiid/figures/haiid_art_pre_heatmap.png)

### haiid — census_post_ai
![haiid_census_post_ai](haiid/figures/haiid_census_post_ai_heatmap.png)

### haiid — census_post_human
![haiid_census_post_human](haiid/figures/haiid_census_post_human_heatmap.png)

### haiid — census_pre
![haiid_census_pre](haiid/figures/haiid_census_pre_heatmap.png)

### haiid — cities_post_ai
![haiid_cities_post_ai](haiid/figures/haiid_cities_post_ai_heatmap.png)

### haiid — cities_post_human
![haiid_cities_post_human](haiid/figures/haiid_cities_post_human_heatmap.png)

### haiid — cities_pre
![haiid_cities_pre](haiid/figures/haiid_cities_pre_heatmap.png)

### haiid — dermatology_post_ai
![haiid_dermatology_post_ai](haiid/figures/haiid_dermatology_post_ai_heatmap.png)

### haiid — dermatology_post_human
![haiid_dermatology_post_human](haiid/figures/haiid_dermatology_post_human_heatmap.png)

### haiid — dermatology_pre
![haiid_dermatology_pre](haiid/figures/haiid_dermatology_pre_heatmap.png)

### haiid — sarcasm_post_ai
![haiid_sarcasm_post_ai](haiid/figures/haiid_sarcasm_post_ai_heatmap.png)

### haiid — sarcasm_post_human
![haiid_sarcasm_post_human](haiid/figures/haiid_sarcasm_post_human_heatmap.png)

### haiid — sarcasm_pre
![haiid_sarcasm_pre](haiid/figures/haiid_sarcasm_pre_heatmap.png)

## hle

### hle — dense
![hle_dense](hle/figures/hle_dense_heatmap.png)

### hle
![hle](hle/figures/hle_heatmap.png)

## iberbench

### iberbench — variety_task
![variety_task](iberbench/figures/variety_task_heatmap.png)

## jailbreakbench

### jailbreakbench — all
![jailbreakbench_all](jailbreakbench/figures/jailbreakbench_all_heatmap.png)

### jailbreakbench — dsn
![jailbreakbench_dsn](jailbreakbench/figures/jailbreakbench_dsn_heatmap.png)

### jailbreakbench — gcg
![jailbreakbench_gcg](jailbreakbench/figures/jailbreakbench_gcg_heatmap.png)

### jailbreakbench — jbc
![jailbreakbench_jbc](jailbreakbench/figures/jailbreakbench_jbc_heatmap.png)

### jailbreakbench — pair
![jailbreakbench_pair](jailbreakbench/figures/jailbreakbench_pair_heatmap.png)

### jailbreakbench — prompt_with_random_search
![jailbreakbench_prompt_with_random_search](jailbreakbench/figures/jailbreakbench_prompt_with_random_search_heatmap.png)

## ko_leaderboard

### ko_leaderboard
![ko_leaderboard](ko_leaderboard/figures/ko_leaderboard_heatmap.png)

## la_leaderboard

### la_leaderboard
![la_leaderboard](la_leaderboard/figures/la_leaderboard_heatmap.png)

## livebench

### livebench — binary
![livebench_binary](livebench/figures/livebench_binary_heatmap.png)

### livebench
![livebench](livebench/figures/livebench_heatmap.png)

## livecodebench

### livecodebench
![livecodebench](livecodebench/figures/livecodebench_heatmap.png)

## matharena

### matharena — aime_2025_II_binary
![matharena_aime_2025_II_binary](matharena/figures/matharena_aime_2025_II_binary_heatmap.png)

### matharena — aime_2025_II
![matharena_aime_2025_II](matharena/figures/matharena_aime_2025_II_heatmap.png)

### matharena — aime_2025_II_raw
![matharena_aime_2025_II_raw](matharena/figures/matharena_aime_2025_II_raw_heatmap.png)

### matharena — aime_2025_I_binary
![matharena_aime_2025_I_binary](matharena/figures/matharena_aime_2025_I_binary_heatmap.png)

### matharena — aime_2025_I
![matharena_aime_2025_I](matharena/figures/matharena_aime_2025_I_heatmap.png)

### matharena — aime_2025_I_raw
![matharena_aime_2025_I_raw](matharena/figures/matharena_aime_2025_I_raw_heatmap.png)

### matharena — aime_2025_binary
![matharena_aime_2025_binary](matharena/figures/matharena_aime_2025_binary_heatmap.png)

### matharena — aime_2025
![matharena_aime_2025](matharena/figures/matharena_aime_2025_heatmap.png)

### matharena — aime_2025_raw
![matharena_aime_2025_raw](matharena/figures/matharena_aime_2025_raw_heatmap.png)

### matharena — aime_2026_I_binary
![matharena_aime_2026_I_binary](matharena/figures/matharena_aime_2026_I_binary_heatmap.png)

### matharena — aime_2026_I
![matharena_aime_2026_I](matharena/figures/matharena_aime_2026_I_heatmap.png)

### matharena — aime_2026_I_raw
![matharena_aime_2026_I_raw](matharena/figures/matharena_aime_2026_I_raw_heatmap.png)

### matharena — aime_2026_binary
![matharena_aime_2026_binary](matharena/figures/matharena_aime_2026_binary_heatmap.png)

### matharena — aime_2026
![matharena_aime_2026](matharena/figures/matharena_aime_2026_heatmap.png)

### matharena — aime_2026_raw
![matharena_aime_2026_raw](matharena/figures/matharena_aime_2026_raw_heatmap.png)

### matharena — aime_combined
![matharena_aime_combined](matharena/figures/matharena_aime_combined_heatmap.png)

### matharena — all_final_answer
![matharena_all_final_answer](matharena/figures/matharena_all_final_answer_heatmap.png)

### matharena — apex_2025_binary
![matharena_apex_2025_binary](matharena/figures/matharena_apex_2025_binary_heatmap.png)

### matharena — apex_2025
![matharena_apex_2025](matharena/figures/matharena_apex_2025_heatmap.png)

### matharena — apex_2025_raw
![matharena_apex_2025_raw](matharena/figures/matharena_apex_2025_raw_heatmap.png)

### matharena — apex_shortlist_binary
![matharena_apex_shortlist_binary](matharena/figures/matharena_apex_shortlist_binary_heatmap.png)

### matharena — apex_shortlist
![matharena_apex_shortlist](matharena/figures/matharena_apex_shortlist_heatmap.png)

### matharena — apex_shortlist_raw
![matharena_apex_shortlist_raw](matharena/figures/matharena_apex_shortlist_raw_heatmap.png)

### matharena — arxivmath_0126_binary
![matharena_arxivmath_0126_binary](matharena/figures/matharena_arxivmath_0126_binary_heatmap.png)

### matharena — arxivmath_0126
![matharena_arxivmath_0126](matharena/figures/matharena_arxivmath_0126_heatmap.png)

### matharena — arxivmath_0126_raw
![matharena_arxivmath_0126_raw](matharena/figures/matharena_arxivmath_0126_raw_heatmap.png)

### matharena — arxivmath_0226_binary
![matharena_arxivmath_0226_binary](matharena/figures/matharena_arxivmath_0226_binary_heatmap.png)

### matharena — arxivmath_0226
![matharena_arxivmath_0226](matharena/figures/matharena_arxivmath_0226_heatmap.png)

### matharena — arxivmath_0226_raw
![matharena_arxivmath_0226_raw](matharena/figures/matharena_arxivmath_0226_raw_heatmap.png)

### matharena — arxivmath_1225_binary
![matharena_arxivmath_1225_binary](matharena/figures/matharena_arxivmath_1225_binary_heatmap.png)

### matharena — arxivmath_1225
![matharena_arxivmath_1225](matharena/figures/matharena_arxivmath_1225_heatmap.png)

### matharena — arxivmath_1225_raw
![matharena_arxivmath_1225_raw](matharena/figures/matharena_arxivmath_1225_raw_heatmap.png)

### matharena — brumo_2025_binary
![matharena_brumo_2025_binary](matharena/figures/matharena_brumo_2025_binary_heatmap.png)

### matharena — brumo_2025
![matharena_brumo_2025](matharena/figures/matharena_brumo_2025_heatmap.png)

### matharena — brumo_2025_raw
![matharena_brumo_2025_raw](matharena/figures/matharena_brumo_2025_raw_heatmap.png)

### matharena — cmimc_2025_binary
![matharena_cmimc_2025_binary](matharena/figures/matharena_cmimc_2025_binary_heatmap.png)

### matharena — cmimc_2025
![matharena_cmimc_2025](matharena/figures/matharena_cmimc_2025_heatmap.png)

### matharena — cmimc_2025_raw
![matharena_cmimc_2025_raw](matharena/figures/matharena_cmimc_2025_raw_heatmap.png)

### matharena — hmmt_feb_2025_binary
![matharena_hmmt_feb_2025_binary](matharena/figures/matharena_hmmt_feb_2025_binary_heatmap.png)

### matharena — hmmt_feb_2025
![matharena_hmmt_feb_2025](matharena/figures/matharena_hmmt_feb_2025_heatmap.png)

### matharena — hmmt_feb_2025_raw
![matharena_hmmt_feb_2025_raw](matharena/figures/matharena_hmmt_feb_2025_raw_heatmap.png)

### matharena — hmmt_feb_2026_binary
![matharena_hmmt_feb_2026_binary](matharena/figures/matharena_hmmt_feb_2026_binary_heatmap.png)

### matharena — hmmt_feb_2026
![matharena_hmmt_feb_2026](matharena/figures/matharena_hmmt_feb_2026_heatmap.png)

### matharena — hmmt_feb_2026_raw
![matharena_hmmt_feb_2026_raw](matharena/figures/matharena_hmmt_feb_2026_raw_heatmap.png)

### matharena — hmmt_nov_2025_binary
![matharena_hmmt_nov_2025_binary](matharena/figures/matharena_hmmt_nov_2025_binary_heatmap.png)

### matharena — hmmt_nov_2025
![matharena_hmmt_nov_2025](matharena/figures/matharena_hmmt_nov_2025_heatmap.png)

### matharena — hmmt_nov_2025_raw
![matharena_hmmt_nov_2025_raw](matharena/figures/matharena_hmmt_nov_2025_raw_heatmap.png)

### matharena — imc_2025_points
![matharena_imc_2025_points](matharena/figures/matharena_imc_2025_points_heatmap.png)

### matharena — imo_2025_points
![matharena_imo_2025_points](matharena/figures/matharena_imo_2025_points_heatmap.png)

### matharena — kangaroo_2025_11_12_binary
![matharena_kangaroo_2025_11_12_binary](matharena/figures/matharena_kangaroo_2025_11_12_binary_heatmap.png)

### matharena — kangaroo_2025_11_12
![matharena_kangaroo_2025_11_12](matharena/figures/matharena_kangaroo_2025_11_12_heatmap.png)

### matharena — kangaroo_2025_11_12_raw
![matharena_kangaroo_2025_11_12_raw](matharena/figures/matharena_kangaroo_2025_11_12_raw_heatmap.png)

### matharena — kangaroo_2025_1_2_binary
![matharena_kangaroo_2025_1_2_binary](matharena/figures/matharena_kangaroo_2025_1_2_binary_heatmap.png)

### matharena — kangaroo_2025_1_2
![matharena_kangaroo_2025_1_2](matharena/figures/matharena_kangaroo_2025_1_2_heatmap.png)

### matharena — kangaroo_2025_1_2_raw
![matharena_kangaroo_2025_1_2_raw](matharena/figures/matharena_kangaroo_2025_1_2_raw_heatmap.png)

### matharena — kangaroo_2025_3_4_binary
![matharena_kangaroo_2025_3_4_binary](matharena/figures/matharena_kangaroo_2025_3_4_binary_heatmap.png)

### matharena — kangaroo_2025_3_4
![matharena_kangaroo_2025_3_4](matharena/figures/matharena_kangaroo_2025_3_4_heatmap.png)

### matharena — kangaroo_2025_3_4_raw
![matharena_kangaroo_2025_3_4_raw](matharena/figures/matharena_kangaroo_2025_3_4_raw_heatmap.png)

### matharena — kangaroo_2025_5_6_binary
![matharena_kangaroo_2025_5_6_binary](matharena/figures/matharena_kangaroo_2025_5_6_binary_heatmap.png)

### matharena — kangaroo_2025_5_6
![matharena_kangaroo_2025_5_6](matharena/figures/matharena_kangaroo_2025_5_6_heatmap.png)

### matharena — kangaroo_2025_5_6_raw
![matharena_kangaroo_2025_5_6_raw](matharena/figures/matharena_kangaroo_2025_5_6_raw_heatmap.png)

### matharena — kangaroo_2025_7_8_binary
![matharena_kangaroo_2025_7_8_binary](matharena/figures/matharena_kangaroo_2025_7_8_binary_heatmap.png)

### matharena — kangaroo_2025_7_8
![matharena_kangaroo_2025_7_8](matharena/figures/matharena_kangaroo_2025_7_8_heatmap.png)

### matharena — kangaroo_2025_7_8_raw
![matharena_kangaroo_2025_7_8_raw](matharena/figures/matharena_kangaroo_2025_7_8_raw_heatmap.png)

### matharena — kangaroo_2025_9_10_binary
![matharena_kangaroo_2025_9_10_binary](matharena/figures/matharena_kangaroo_2025_9_10_binary_heatmap.png)

### matharena — kangaroo_2025_9_10
![matharena_kangaroo_2025_9_10](matharena/figures/matharena_kangaroo_2025_9_10_heatmap.png)

### matharena — kangaroo_2025_9_10_raw
![matharena_kangaroo_2025_9_10_raw](matharena/figures/matharena_kangaroo_2025_9_10_raw_heatmap.png)

### matharena — miklos_2025_points
![matharena_miklos_2025_points](matharena/figures/matharena_miklos_2025_points_heatmap.png)

### matharena — putnam_2025_points
![matharena_putnam_2025_points](matharena/figures/matharena_putnam_2025_points_heatmap.png)

### matharena — smt_2025_binary
![matharena_smt_2025_binary](matharena/figures/matharena_smt_2025_binary_heatmap.png)

### matharena — smt_2025
![matharena_smt_2025](matharena/figures/matharena_smt_2025_heatmap.png)

### matharena — smt_2025_raw
![matharena_smt_2025_raw](matharena/figures/matharena_smt_2025_raw_heatmap.png)

### matharena — usamo_2025_points
![matharena_usamo_2025_points](matharena/figures/matharena_usamo_2025_points_heatmap.png)

## metr_early2025

### metr_early2025 — ai
![metr_early2025_ai](metr_early2025/figures/metr_early2025_ai_heatmap.png)

### metr_early2025 — no_ai
![metr_early2025_no_ai](metr_early2025/figures/metr_early2025_no_ai_heatmap.png)

## metr_late2025

### metr_late2025 — ai-allowed
![metr_late2025_ai-allowed](metr_late2025/figures/metr_late2025_ai-allowed_heatmap.png)

### metr_late2025 — ai-disallowed
![metr_late2025_ai-disallowed](metr_late2025/figures/metr_late2025_ai-disallowed_heatmap.png)

## mmlupro

### mmlupro — category
![mmlupro_category](mmlupro/figures/mmlupro_category_heatmap.png)

### mmlupro
![mmlupro](mmlupro/figures/mmlupro_heatmap.png)

## osworld

### osworld — binary
![osworld_binary](osworld/figures/osworld_binary_heatmap.png)

### osworld
![osworld](osworld/figures/osworld_heatmap.png)

## pt_leaderboard

### pt_leaderboard
![pt_leaderboard](pt_leaderboard/figures/pt_leaderboard_heatmap.png)

## sib200

### sib200 — script_accuracy
![script_accuracy](sib200/figures/script_accuracy_heatmap.png)

## swebench

### swebench
![swebench](swebench/figures/swebench_heatmap.png)

## swebench_full

### swebench_full
![swebench_full](swebench_full/figures/swebench_full_heatmap.png)

## swebench_java

### swebench_java
![swebench_java](swebench_java/figures/swebench_java_heatmap.png)

### swebench_java — multi_swebench_java
![swebench_java_multi_swebench_java](swebench_java/figures/swebench_java_multi_swebench_java_heatmap.png)

### swebench_java — swebench_multilingual_java
![swebench_java_swebench_multilingual_java](swebench_java/figures/swebench_java_swebench_multilingual_java_heatmap.png)

## swebench_multilingual

### swebench_multilingual
![swebench_multilingual](swebench_multilingual/figures/swebench_multilingual_heatmap.png)

### swebench_multilingual — multi_swebench_c++
![swebench_multilingual_multi_swebench_c++](swebench_multilingual/figures/swebench_multilingual_multi_swebench_c++_heatmap.png)

### swebench_multilingual — multi_swebench_c
![swebench_multilingual_multi_swebench_c](swebench_multilingual/figures/swebench_multilingual_multi_swebench_c_heatmap.png)

### swebench_multilingual — multi_swebench_go
![swebench_multilingual_multi_swebench_go](swebench_multilingual/figures/swebench_multilingual_multi_swebench_go_heatmap.png)

### swebench_multilingual — multi_swebench
![swebench_multilingual_multi_swebench](swebench_multilingual/figures/swebench_multilingual_multi_swebench_heatmap.png)

### swebench_multilingual — multi_swebench_java
![swebench_multilingual_multi_swebench_java](swebench_multilingual/figures/swebench_multilingual_multi_swebench_java_heatmap.png)

### swebench_multilingual — multi_swebench_javascript
![swebench_multilingual_multi_swebench_javascript](swebench_multilingual/figures/swebench_multilingual_multi_swebench_javascript_heatmap.png)

### swebench_multilingual — multi_swebench_python
![swebench_multilingual_multi_swebench_python](swebench_multilingual/figures/swebench_multilingual_multi_swebench_python_heatmap.png)

### swebench_multilingual — multi_swebench_rust
![swebench_multilingual_multi_swebench_rust](swebench_multilingual/figures/swebench_multilingual_multi_swebench_rust_heatmap.png)

### swebench_multilingual — multi_swebench_typescript
![swebench_multilingual_multi_swebench_typescript](swebench_multilingual/figures/swebench_multilingual_multi_swebench_typescript_heatmap.png)

## swepolybench

### swepolybench — full
![swepolybench_full](swepolybench/figures/swepolybench_full_heatmap.png)

### swepolybench — verified
![swepolybench_verified](swepolybench/figures/swepolybench_verified_heatmap.png)

## taubench

### taubench — hal_airline
![taubench_hal_airline](taubench/figures/taubench_hal_airline_heatmap.png)

### taubench — retail
![taubench_retail](taubench/figures/taubench_retail_heatmap.png)

### taubench — v1_airline
![taubench_v1_airline](taubench/figures/taubench_v1_airline_heatmap.png)

### taubench — v2_airline
![taubench_v2_airline](taubench/figures/taubench_v2_airline_heatmap.png)

## thai_leaderboard

### thai_leaderboard
![thai_leaderboard](thai_leaderboard/figures/thai_leaderboard_heatmap.png)

## theagentcompany

### theagentcompany — binary
![theagentcompany_binary](theagentcompany/figures/theagentcompany_binary_heatmap.png)

### theagentcompany
![theagentcompany](theagentcompany/figures/theagentcompany_heatmap.png)

## toolbench

### toolbench — G1_category
![toolbench_G1_category](toolbench/figures/toolbench_G1_category_heatmap.png)

### toolbench — G1_instruction
![toolbench_G1_instruction](toolbench/figures/toolbench_G1_instruction_heatmap.png)

### toolbench — G1_tool
![toolbench_G1_tool](toolbench/figures/toolbench_G1_tool_heatmap.png)

### toolbench — G2_category
![toolbench_G2_category](toolbench/figures/toolbench_G2_category_heatmap.png)

### toolbench — G2_instruction
![toolbench_G2_instruction](toolbench/figures/toolbench_G2_instruction_heatmap.png)

### toolbench — G3_instruction
![toolbench_G3_instruction](toolbench/figures/toolbench_G3_instruction_heatmap.png)

### toolbench
![toolbench](toolbench/figures/toolbench_heatmap.png)

## visualwebarena

### visualwebarena
![visualwebarena](visualwebarena/figures/visualwebarena_heatmap.png)

## workarena

### workarena — agentrewardbench
![workarena_agentrewardbench](workarena/figures/workarena_agentrewardbench_heatmap.png)

### workarena
![workarena](workarena/figures/workarena_heatmap.png)

### workarena — leaderboard
![workarena_leaderboard](workarena/figures/workarena_leaderboard_heatmap.png)

### workarena — paper_l1_categories
![workarena_paper_l1_categories](workarena/figures/workarena_paper_l1_categories_heatmap.png)
