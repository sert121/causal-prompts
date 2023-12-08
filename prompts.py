import openai
import numpy as np
import dotenv, os
import glob
import re, time
from openai import OpenAI
from tqdm import tqdm

from helpers import generate_pairs

dotenv.load_dotenv()


def construct_arctic_dataset():
    """loads/constructs the arctic dataset
    as per the paper outlines it.
    Creates a ground truth graph based on the relationships
    outlined in the paper.

    Returns:
        causal_pair_graph (dict) :  Graph for the arctic dataset
        variable_mappings (dict) :  Mapping from variable name to description
        nodes (list) : List of nodes in the graph

    """
    nodes = [
        "LW",
        "SW",
        "CC",
        "CW",
        "GH",
        "RH",
        "SLP",
        "u10m",
        "v10m",
        "Sea_ice",
        "Precip",
        "HFLX",
    ]
    variable_mappings = {
        "GH": "Geopotential heights averaged from 200 hPa, 500 hPa, and 850 hPa",
        "RH": "Relative humidity averaged from 1,000–300 hPa",
        "SLP": "Sea level pressure",
        "u10m": "Zonal (u-component) wind at 10 m",
        "v10m": "Meridional (v-component) wind at 10 m",
        "HFLX": "Sensible plus latent heat flux",
        "Precip": "Total precipitation",
        "CC": "Total cloud cover",
        "CW": "Total cloud water path",
        "SW": "Net shortwave flux at the surface",
        "LW": "Net longwave flux at the surface",
        "Sea_ice": "Sea ice extent in the Northern Hemisphere",
    }

    causal_pair_graph = {
        "LW": ["Sea_ice"],
        "SW": ["Sea_ice"],
        "CC": ["LW", "SW", "RH", "Precip", "Sea_ice", "HFLX"],
        "CW": ["Precip", "HLFX", "SW", "LW", "RH"],
        "GH": ["LW", "RH", "SLP"],
        "RH": ["CC", "CW", "LW", "Precip"],
        "SLP": ["RH", "GH", "Sea_ice", "HFLX", "u10m", "v10m"],
        "u10m": ["Sea_ice", "HFLX"],
        "v10m": ["Sea_ice", "HFLX"],
        "Sea_ice": ["SW", "u10", "SLP", "v10" "HFLX"],
        "HFLX": ["CW", "CC", "u10", "v10", "SLP", "Precip", "Sea_ice"],
        "Precip": ["CW", "CC", "Sea_ice", "RH", "HFLX", "LW"],
    }

    return causal_pair_graph, variable_mappings, nodes


def construct_prompt(option_a, option_b):
    PROMPT = f"""
You are a helpful assistant to experts in artic sea ice research and atmospheric science.
Which cause-and-effect relationship is more likely?
A. {option_a} causes {option_b}
B. {option_b} causes {option_a}
C. Neither
Let’s work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B/C</Answer>.
    """
    return PROMPT


def answer_query(sentence):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert on understand arctic sea and atmoshperic science.",
            },
            {
                "role": "user",
                "content": f"{sentence}",
            },
        ],
        model="gpt-3.5-turbo",
    )
    label = chat_completion.choices[0].message.content
    return label


def extract_answer(answer):
    answer = re.findall(r"<Answer>(.*?)</Answer>", answer)
    answer = answer[0]
    return answer


def resolve_cause_effect(possible_pairs, variable_mappings):
    """_summary_

    Args:
        possible_pairs
        variable_mappings
    """
    final_graph = {key: [] for key in variable_mappings.keys()}
    for pair in tqdm(possible_pairs[:2]):
        option_a, option_b = pair
        A, B = variable_mappings[option_a], variable_mappings[option_b]
        prompt = construct_prompt(A, B)
        answer = answer_query(prompt)
        print(answer)
        true_val = extract_answer(answer)
        if true_val == "C":
            continue
        final_graph[option_a].append(option_b) if true_val == "A" else final_graph[
            option_b
        ].append(option_a)


def run_pipeline():
    _, variable_mappings, nodes = construct_arctic_dataset()
    possible_pairs = generate_pairs(nodes)
    resolve_cause_effect(possible_pairs, variable_mappings)


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key,
    )
    run_pipeline()
