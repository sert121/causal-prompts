import openai
import numpy as np
import dotenv, os
import glob
import re, time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from helpers import generate_pairs
import json
dotenv.load_dotenv()



def save_dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def construct_arctic_dataset():
    """loads/constructs the arctic dataset
    as per the paper outlines it.
    Creates a ground truth graph based on the relationships
    outlined in the paper. https://www.frontiersin.org/files/Articles/642182/fdata-04-642182-HTML-r1/image_m/fdata-04-642182-g001.jpg

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
        "Sea_ice": ["SW", "u10m", "SLP", "v10m","HFLX"],
        "HFLX": ["CW", "CC", "u10m", "v10m", "SLP", "Precip", "Sea_ice"],
        "Precip": ["CW", "CC", "Sea_ice", "RH", "HFLX", "LW"],
    }

    return causal_pair_graph, variable_mappings, nodes


def construct_prompt(option_a, option_b):
    """
    Constructs a prompt for the GPT-3 model to answer
    the cause-effect relationship between two variables.
    Args:
        option_a (str) : First variable
        option_b (str) : Second variable
    Returns:
        PROMPT (str) : Prompt for the language model
    """

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
    """Answers a query based on the prompt
    Args:
        sentence (str) : Prompt to answer
    Returns:
        label (str) : Label for the cause-effect relationship
    """
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
    """Extracts the answer from the GPT-3 response
    Args:
        answer (str) : GPT-3 response
    Returns:
        answer (str) : Answer to the cause-effect relationship
    """
    answer = re.findall(r"<Answer>(.*?)</Answer>", answer)
    answer = answer[0]
    return answer


def resolve_cause_effect(possible_pairs, variable_mappings, causal_pair_graph):
    """Resolves the cause-effect relationship between
    two variables
    Args:
        possible_pairs (list) : List of possible pairs
        variable_mappings (dict) : Mapping from variable name to description
        causal_pair_graph (dict) : Graph for the arctic dataset
    Returns:
        final_graph (dict) : Graph of the cause-effect relationships
    """
    final_graph = {key: [] for key in variable_mappings.keys()}
    # calculate the answer for each pair
    pair_wise_answers = {}
    ground_truth = []
    for pair in tqdm(possible_pairs[:]):
        option_a, option_b = pair
        if option_b in causal_pair_graph[option_a]:
            ground_truth.append("A")
        elif option_a in causal_pair_graph[option_b]:
            ground_truth.append("B")
        else:
            ground_truth.append("None")
            
    predicted_labels = []

    # generated_answers = {}
    for pair in tqdm(possible_pairs[:]):
        error_indexes = []
        option_a, option_b = pair
        A, B = variable_mappings[option_a], variable_mappings[option_b]
        prompt = construct_prompt(A, B)
        answer = answer_query(prompt)
        pair_wise_answers[f"{pair}"] = answer

        # save in a json file
        

        time.sleep(1)
        print(answer)
        
        try:
            true_val = extract_answer(answer)
        except Exception as e:
            pair_wise_answers[f"{pair}"] = answer
            print(f"Error in extracting answer from pair {pair}", e)
            error_indexes.append(pair)

            # true_val = "C"
        if true_val == "C":
            predicted_labels.append("None")
            continue
        elif true_val == "A":
            predicted_labels.append("A")
            final_graph[option_a].append(option_b)
        elif true_val == "B":
            predicted_labels.append("B")
            final_graph[option_b].append(option_a)
        else:    
            predicted_labels.append("NA")

    try:
        print(accuracy_score(np.array(ground_truth), np.array(predicted_labels)))
    except Exception as e:
        print("Accuracy could not be calculated", e)
    save_dict_to_json(pair_wise_answers, "pair_wise_answers.json")
    save_dict_to_json(final_graph, "generated_answers.json")
    return final_graph


def run_pipeline():
    """Runs the pipeline for the arctic dataset
    """
    causal_pair_graph, variable_mappings, nodes = construct_arctic_dataset()
    possible_pairs = generate_pairs(nodes)
    generated_graph = resolve_cause_effect(possible_pairs, variable_mappings,causal_pair_graph)

def normalized_hamming_distance(prediction, target):
  '''
  prediction and target are edge lists
  calculate the normalized hamming distance

  For a graph with m nodes, the distance is given by ∑m i,j=1 1 m2 1Gij 6=G′ ij , 
  the number of edges that are present in one graph but not the other, 
  divided by the total number of all possible edges.
  '''
  prediction = set(prediction)
  target = set(target)
  total_nodes = set()
  for i,j in target:
    total_nodes.add(i)
    total_nodes.add(j)
  no_overlap = len(prediction.union(target)) - len(prediction.intersection(target))
  return no_overlap / (len(total_nodes) ** 2)

def eval_pipeline():
    generated_graph = read_json("generated_answers.json")
    og_graph ,_,nodes = construct_arctic_dataset()
    possible_pairs = generate_pairs(nodes)
    generated_pairs = []
    original_pairs = []
    for pair in possible_pairs:
        option_a,option_b = pair
        if option_b in generated_graph[option_a]:
            generated_pairs.append((option_a, option_b))
        elif option_a in generated_graph[option_b]:
            generated_pairs.append((option_b, option_a))
    
    for pair in possible_pairs:
        option_a,option_b = pair
        if option_b in og_graph[option_a]:
            original_pairs.append((option_a, option_b))
        elif option_a in og_graph[option_b]:
            original_pairs.append((option_b, option_a))
    correct = 0
    for i in original_pairs:
        if i in generated_pairs:
            correct+=1        

    distance = normalized_hamming_distance(generated_pairs, original_pairs)
    print("Normalized Hamming Distance: ", distance)

    precision = correct/len(generated_pairs)
    recall = correct/len(original_pairs)
    accuracy_score = correct/len(original_pairs)
    print("Precision: ", precision)
    print("Recall: ", recall)


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key,
    )

    run_pipeline()
    # eval_pipeline()