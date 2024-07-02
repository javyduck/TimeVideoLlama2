import argparse
import torch

from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.model.builder import load_pretrained_model
from videollama2.utils import disable_torch_init
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, KeywordsStoppingCriteria, process_video, expand2square
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from decord import VideoReader, cpu
import os
import json
from tqdm import tqdm

default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
default_mm_start_token =  DEFAULT_MMODAL_START_TOKEN["VIDEO"]
default_mm_end_token = DEFAULT_MMODAL_END_TOKEN["VIDEO"]
modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]

def main(args):
    # Questions:

    json_file = args.input
    os.makedirs(args.output, exist_ok=True)
    out_json_paths = [f"{args.output}/BDDX_Test_pred_{cap}.json" for cap in ['action','justification','control_signal']]
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device=args.device)
    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    # print(model, tokenizer, processor)
    # image_processor = processor['image']

    conv_mode = "driving"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()


    # gt
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Pred
    out_jsons = [[],[],[]]
    
    for item in tqdm(data):
        q1, q2, q3 = item["conversations"][0]["value"], item["conversations"][2]["value"], item["conversations"][4]["value"]
        conv.messages.clear()
        roles = conv.roles
            
        vps, vid = item["video"], item['id']
        if type(vps) != list:
            vps = [vps]
            
        video_paths = [os.path.join("./video_process/BDDX_Test/",vp) for vp in vps]

        
        video_tensor = process_video(video_paths[0], processor, aspect_ratio='pad', sample_scheme='uniform', num_frames=num_frames)
        if type(video_tensor) is list:
            tensor = [[video.to(model.device, dtype=torch.float16) for video in video_tensor]]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)
        key = ['video']
        
        inst_answers = []
        for qid, question in enumerate([q1,q2,q3]):
            # print(question)
            inp = question
            
            if vps is not None:
                # First Message
#                 inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
                inp = inp
                conv.append_message(conv.roles[0], inp)
                video = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#             print(len(tensor), key)
            # import pdb;pdb.set_trace()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images_or_videos=video_tensor.unsqueeze(0).half(),
                    modal_list=['video'],
                    do_sample=False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            conv.messages[-1][-1] = outputs
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            
            inst_pred = {
                "image_id":vid,
                "caption":outputs
            }

            out_jsons[qid].append(inst_pred)
        # import pdb; pdb.set_trace()
        # break
    
        # Save separate json for action and justification
    for i in range(3):
        with open(out_json_paths[i],"w") as of:
            json.dump(out_jsons[i], of, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
