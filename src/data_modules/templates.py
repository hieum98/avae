from typing import List, Union

from pydantic import BaseModel


def apply_qwen2_template_for_rep_learning(text: Union[List[str], str]) -> Union[List[str], str]:
    start_token = '<|im_start|>'
    end_token = '<|im_end|>'
    if isinstance(text, list):
        return [f"{start_token}{txt}{end_token}" for txt in text]
    elif isinstance(text, str):
        return f"{start_token}{text}{end_token}"
    else:
        raise ValueError(f"Input type not recognized: {type(text)}")
    
def apply_template_for_rep_learning(text: Union[List[str], str], model_type: str) -> Union[List[str], str]:
    # TODO: Currently do nothing, just return the text. Future work should implement the template for each model type.
    # if model_type == 'qwen2':
    #     return apply_qwen2_template_for_rep_learning(text)
    # else:
    #     print(f"Warning: model_type: {model_type} not recognized. Using original text.")
    #     return text
    return text

def tokenize_example(
        input: Union[List[str], str], 
        tokenizer, 
        max_seq_length, 
        **kwargs
        ):
    tokenized_txt = tokenizer(
        input,
        max_length=max_seq_length,
        truncation=kwargs.get('truncation', True),
        padding=kwargs.get('padding', 'longest'),
        return_tensors=kwargs.get('return_tensors', 'pt'),
        add_special_tokens=kwargs.get('add_special_tokens', False),
    )
    return tokenized_txt


GENERATE_STYLE_COUNTERFACTUAL_PROMPT = "You are a rewriter. You should analyze the writing style of the text \n"\
                                        "And then, rewite it into a substantively different style while:\n"\
                                        "1. Maintaining the original meaning and context and factual accuracy.\n"\
                                        "2. Using contrasting stylistic features from the original analysis.\n"\
                                        "3. Avoiding any grammatical errors or awkward phrasing.\n"\
                                        "4. Ensuring the rewrited text is coherent and flows naturally.\n"\
                                        "5. Avoiding style-specific keywords (e.g., don't say 'make it casual').\n"\
                                        "6. The rewrited text should be same langauge as the original text.\n"\
                                        "Please ONLY generate the output in the JSON format with these keys: \n"\
                                        '- "rewrited_text": the rewrited text\n'\
                                        '- "style_comparison": Your analysis explaining why the rewrited text appear to be written by the different authors."'
STYLE_COUNTERFACTUAL_IN_CONTEXT_EXAMPLES = [
        {
            "input_text": "The quarterly earnings report indicates a 12.3 points revenue increase attributable to emerging market expansions.",
            "rewrited_text": "Our money numbers just popped off — we're rolling in 12% more cash thanks to hitting up new countries! Cha-ching!",
            "style_comparison": "Original uses formal corporate jargon ('quarterly earnings report,' 'attributable to') with passive voice and technical percentages. Transformation employs financial slang ('money numbers,' 'popped off'), enthusiastic interjections ('Cha-ching!'), and active voice, replacing institutional phrasing with celebratory colloquialisms while maintaining numerical accuracy."
        },
        {
            "input_text": "Mitochondria utilize oxidative phosphorylation to synthesize adenosine triphosphate through proton gradient-driven chemiosmosis.",
            "rewrited_text": "Those tiny cell batteries make energy packets by shuffling charged particles across their membranes — it's like nature's power plant at work!",
            "style_comparison": "Original features dense scientific terminology and complex biochemical processes. Transformation uses accessible analogies ('tiny cell batteries,' 'nature's power plant'), metaphorical language, and layman's terms while preserving the essential mechanism description through concrete imagery rather than technical terms."
        },
        {
            "input_text": "Kindly refrain from auditory disruptions during nocturnal hours to facilitate community repose.",
            "rewrited_text": "Yo, keep it down after dark — we're all trying to catch some Z's here!",
            "style_comparison": "Original employs archaic polite phrasing ('Kindly refrain,' 'facilitate community repose') with formal nominalizations. Transformation uses contemporary slang ('Yo,' 'catch some Z's'), direct imperative structure, and neighborhood vernacular to convey the same request with streetwise immediacy rather than institutional decorum."
        }
    ]
class StyleTransferReply(BaseModel):
    rewrited_text: str
    style_comparison: str


STYLE_COMPRARITOR_SYSTEM_PROMPT = "You are a literary style analyst with expertise in authorship attribution. " \
"Your task is to analyze two text samples and determine whether they were written by the same author based on stylistic features."
STYLE_COMPRARITOR_USER_PROMPT = """Given two text samples and a label indicating whether they were written by the same author or not, provide a stylistic analysis explaining why the attribution is correct. 

Your analysis should be concise but thorough, highlighting the most significant stylistic markers that support the given attribution.

Output your analysis in JSON format as follows:
  - 'style_comparison': 'Your analysis explaining why the texts appear to be written by the same author or different authors.'

Text 1: {text1}

Text 2: {text2}

Label: {label}
"""

class StyleComparisonReply(BaseModel):
    style_comparison: str


CONTENT_COMPARITOR_SYSTEM_PROMPT = "You are an expert content analyst with exceptional skills in identifying semantic similarities and differences between texts. Your task is to analyze two provided texts and determine if they express the same core content, regardless of stylistic differences."
CONTENT_COMPARITOR_USER_PROMPT = """Given two text samples, you should:
1. Carefully identify the main ideas, key points, and central arguments in each text
2. Look beyond surface-level wording and focus on the underlying meaning
3. Determine if both texts convey the same essential information and purpose
4. Ignore differences in:
   - Writing style
   - Vocabulary choice
   - Sentence structure
   - Length
   - Examples used (if they serve the same purpose)
5. Focus on whether the core message and information remain consistent

Provide your analysis in valid JSON format with exactly two fields:
1. 'content_comparison': A concise explanation justifying your determination, highlighting key similarities or differences in content
2. 'determination': Either 'same content' or 'different content'

Text 1: {text1}

Text 2: {text2}
"""

class ContentComparisonReply(BaseModel):
    content_comparison: str
    determination: str


STYLE_REP_COMPARITOR_PROMPT = """Given two style representations, determine if they are written by the same author or not. Your analysis should focus on stylistic features.

Your analysis should be concise but thorough, highlighting the most significant stylistic markers that support the given attribution.

Provide your analysis in valid JSON format with exactly two fields:
1. 'determination': 'same author' or 'different author'
2. 'explaination': Your analysis explaining why the texts appear to be written by the same author or different authors.

The style representations for Text 1, Text 2, respectively, are:
Text 1's style representation: {text1}
Text 2's style representation: {text2}
"""

class StyleRepComparisonReply(BaseModel):
    determination: str
    explaination: str

CONTENT_REP_COMPARITOR_PROMPT = """Given two content representations, determine if they express the same core content, regardless of stylistic differences. Your analysis should focus on:
1. Carefully identify the main ideas, key points, and central arguments in each text
2. Look beyond surface-level wording and focus on the underlying meaning
3. Determine if both texts convey the same essential information and purpose
4. Ignore differences in:
   - Writing style
   - Vocabulary choice
   - Sentence structure
   - Length
   - Examples used (if they serve the same purpose)
5. Focus on whether the core message and information remain consistent

Provide your analysis in valid JSON format with exactly two fields:
1. 'explaination': A concise explanation justifying your determination, highlighting key similarities or differences in content
2. 'determination': Either 'same content' or 'different content'

The content representations for Text 1, Text 2, respectively, are:
Text 1's content representation: {text1}
Text 2's content representation: {text2}
"""

class ContentRepComparisonReply(BaseModel):
    explaination: str
    determination: str


RECONSTRUCT_PROMPT = """Given the style representation and content representation of a text, reconstruct the original text.
Style representation: {style_rep}
Content representation: {content_rep}
"""

