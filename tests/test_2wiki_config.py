from mirage.config import load_config


def test_2wiki_configs_are_scoped_to_retrieval_side_families():
    pilot = load_config("configs/experiments/E1_2wiki_pilot.yaml")
    full = load_config("configs/experiments/E4_2wiki_v3.yaml")
    assert pilot.dataset_name == "framolfese/2WikiMultihopQA"
    assert full.clean_examples == 1000
    assert full.max_examples == full.natural_examples == 1000
    assert full.injector_families == ["query_retrieval", "source_degradation"]
    assert full.answer_top_k < full.top_k
