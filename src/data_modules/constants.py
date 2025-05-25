ALL_PRETRAIN_DATA = [
    'Hieuman/gmane_dataset-doc', #
    'Hieuman/realnews-doc',
    'Hieuman/food.com-doc',
    'Hieuman/goodreads-doc',
    'Hieuman/reddit-doc', # Hieuman/MUD, Hieuman/reddit_dump
    'Hieuman/yelp_review-doc',
    'Hieuman/amazon_reviews-doc',
    'Hieuman/STX-doc',
    'Hieuman/HNI-doc',
    'Hieuman/nyt_comments-doc',
    'Hieuman/blog_authorship-doc',
    'Hieuman/hiatus-imdb-doc',
    'Hieuman/twitter-doc',
    'Hieuman/wiki_en-doc',
]

ALL_DISENT_DATA = [
    'Hieuman/reddit_dumpavae',
    'Hieuman/HNIavae',
    'Hieuman/STXavae',
    'Hieuman/amazon_reviewsavae',
    'Hieuman/wiki_en_smallavae',
    'Hieuman/hiatus-imdbavae',
    'Hieuman/blog_authorshipavae',
    'Hieuman/nyt_commentsavae',
    'Hieuman/goodreadsavae',
    'Hieuman/realnews_smallavae',
    'Hieuman/yelp_reviewavae'
#     'Hieuman/avae',
]

HRS_DIR = 'data/HRS/TA1'
HRS_PATHS = {
    'HRS1.1': {
            'resample_queries': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.1/data/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.1/data/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.1/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.1_TA1_groundtruth.jsonl'
    },
    'HRS1.2': {
            'resample_queries': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.2/data/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.2/data/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.2/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.2_TA1_groundtruth.jsonl'
    },
    'HRS1.3': {
            'resample_queries': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.3/data/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.3/data/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.3/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.3_TA1_groundtruth.jsonl'
    },
    'HRS1.4': {
            'resample_queries': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.4/data/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.4/data/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.4/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.4_TA1_groundtruth.jsonl'
    },
    'HRS1.5': {
            'resample_queries': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.5/data/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.5/data/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long_sample-0_perGenre-HRS1.5/groundtruth/HRS1_english_long_sample-0_perGenre-HRS1.5_TA1_groundtruth.jsonl'
    },
    'en_cross_genre_long': {
            'resample_queries': f'{HRS_DIR}/HRS1_english_long_sample-0_crossGenre/data/HRS1_english_long_sample-0_crossGenre_TA1_input_queries.jsonl',
            'resample_candidates': f'{HRS_DIR}/HRS1_english_long_sample-0_crossGenre/data/HRS1_english_long_sample-0_crossGenre_TA1_input_candidates.jsonl',
            'ground_truth': f'{HRS_DIR}/HRS1_english_long_sample-0_crossGenre/groundtruth/HRS1_english_long_sample-0_crossGenre_TA1_groundtruth.jsonl'
    },
}


AMAZON_REVIEWS_PATHS = {
    'queries': 'data/amazon_reviews/validation_queries.jsonl',
    'candidates': 'data/amazon_reviews/validation_targets.jsonl',
}

MUD_PATHS = {
    'queries': 'data/MUD/test_queries.jsonl',
    'candidates': 'data/MUD/test_targets.jsonl',
}

PAN20_PATHS = {
    'queries': 'data/PAN/pan_20_queries_raw.jsonl',
    'candidates': 'data/PAN/pan_20_targets_raw.jsonl',
}

PAN21_PATHS = {
    'queries': 'data/PAN/pan_21_queries_raw.jsonl',
    'candidates': 'data/PAN/pan_21_targets_raw.jsonl',
}

