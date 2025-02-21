import argparse
import os
import sys
import time

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.video import VideoChunkLoader

from ov_lvm_wrapper import OVMiniCPMV26Worker


def output_handler(text: str,
                   filename: str = '',
                   mode: str = 'w',
                   verbose: bool = True):
    # Print to terminal
    if verbose:
        print(text)

    # Write to file, if requested
    if filename != '':
        with open(filename, mode) as FH:
            print(text, file=FH)


if __name__ == '__main__':
    # Parse inputs
    parser_txt = "Generate video summarization using Langchanin, OpenVINO-genai, and MiniCPM-V-2_6."
    parser = argparse.ArgumentParser(parser_txt)
    parser.add_argument("video_file", type=str,
                        help='Path to video you want to summarize.')
    parser.add_argument("model_dir", type=str,
                        help="Path to openvino-genai optimized model")
    parser.add_argument("-p", "--prompt", type=str,
                        help="Text prompt. By default set to: `Please summarize this video.`",
                        default="Please summarize this video.")
    parser.add_argument("-d", "--device", type=str,
                        help="Target device for running ov MiniCPM-v-2_6",
                        default="CPU")
    parser.add_argument("-t", "--max_new_tokens", type=int,
                        help="Maximum number of tokens to be generated.",
                        default=500)
    parser.add_argument("-f", "--max_num_frames", type=int,
                        help="Maximum number of frames to be sampled per chunk for inference. Set to a smaller number if OOM.",
                        default=32)
    parser.add_argument("-c", "--chunk_duration", type=int,
                        help="Maximum length in seconds for each chunk of video.",
                        default=30)
    parser.add_argument("-v", "--chunk_overlap", type=int,
                        help="Overlap in seconds beteen chunks of input video.",
                        default=2)
    parser.add_argument("-r", "--resolution", type=int, nargs=2,
                        help="Desired spatial resolution of input video if different than original. Width x Height")
    parser.add_argument("-o", "--outfile", type=str,
                        help="File to write generated text.", default='')

    tot_st_time = time.time()
    args = parser.parse_args()
    if not os.path.exists(args.video_file):
        print(f"{args.video_file} does not exist.")
        exit()

    # Create template for inputs
    prompt = PromptTemplate(
        input_variables=["video", "question"],
        template="{video},{question}"
    )

    # Wrap OpenVINO-GenAI optimized model in custom langchain wrapper
    resolution = [] if not args.resolution else args.resolution
    ov_minicpm = OVMiniCPMV26Worker(model_dir=args.model_dir,
                                    device=args.device,
                                    max_new_tokens=args.max_new_tokens,
                                    max_num_frames=args.max_num_frames,
                                    resolution=resolution)

    # Create pipeline and invoke
    chain = prompt | ov_minicpm

    # Initialize video chunk loader
    loader = VideoChunkLoader(
        video_path=args.video_file,
        chunking_mechanism="sliding_window",
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap)

    # Start log
    output_handler("python " + " ".join(sys.argv),
                   filename=args.outfile, mode='w',
                   verbose=False)

    # Loop through docs and generate chunk summaries    
    chunk_summaries = []
    for doc in loader.lazy_load():
        # Log metadata
        output_handler(str(f"Chunk Metadata: {doc.metadata}"),
                       filename=args.outfile, mode='a')
        output_handler(str(f"Chunk Content: {doc.page_content}"),
                       filename=args.outfile, mode='a')

        # Generate sumarries
        chunk_st_time = time.time()
        inputs = {"video": doc.metadata['chunk_path'], "question": args.prompt}
        output = chain.invoke(inputs)

        # Log output
        output_handler(output, filename=args.outfile, mode='a', verbose=False)
        chunk_summaries.append(
            f"Start time: {doc.metadata['start_time']} End time: {doc.metadata['end_time']}\n" + output)
        output_handler("\nChunk Inference time: {} sec\n".format(time.time() - chunk_st_time), filename=args.outfile,
                       mode='a')

    # Summarize the full video, using the subsections summaries from each chunk
    overall_summ_st_time = time.time()
    full_summ_prompt = 'The following are summaries of subsections of a video. Each subsection summary is separated by the delimiter ">|<". Each subsection summary will start with the start and end timestamps of the subsection relative to the full video. Please create a summary of the overall video, highlighting all important information, including timestamps:\n\n{}'
    inputs = {"video": '', "question": full_summ_prompt.format("\n>|<\n".join(chunk_summaries))}
    output = chain.invoke(inputs)
    output_handler("\nOverall video summary inference time: {} sec\n".format(time.time() - overall_summ_st_time),
                   filename=args.outfile, mode='a')
    output_handler("\nTotal Inference time: {} sec\n".format(time.time() - tot_st_time), filename=args.outfile,
                   mode='a')
    output_handler(output, filename=args.outfile, mode='a', verbose=False)
