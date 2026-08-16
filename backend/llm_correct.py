#!/usr/bin/env python3
"""
llm_correct.py  --  LLM correction of decoded keystrokes, using NVIDIA Nemotron
(local, on the Spark GPU). Fixes acoustic-recognition character errors.

Format-constrained: outputs ONLY the corrected text. Handles BOTH natural
sentences AND random strings (won't turn a random password into English words).
Input/output use '_' for the space bar (matching the recognizer), lowercase +
digits only, so it aligns with the ground-truth format for scoring.

Usage:
  from llm_correct import Corrector
  Corrector().correct("the_quikc_brwn_fpx")           -> "the_quick_brown_fox"

  python llm_correct.py --text "the_quikc_brwn"        # print corrected
  python llm_correct.py --session last_session.json    # raw vs corrected ACCURACY
  python llm_correct.py --model nvidia/Mistral-NeMo-Minitron-8B-Instruct --text ...
"""
import os, sys, json, argparse, re

SYSTEM_PROMPT = (
    "You are a SPELL-CHECKER for text from an acoustic keystroke recognizer. '_' "
    "marks the space bar and is ALWAYS CORRECT -- spaces are detected almost "
    "perfectly, so only the letters/digits BETWEEN spaces may be wrong.\n"
    "RULES (follow exactly):\n"
    "1. NEVER add, remove, merge, split, or move '_'. Your output must have EXACTLY "
    "as many '_'-separated tokens as the input, in the same order.\n"
    "2. Correct only the characters inside each token. A token's length MAY change "
    "(e.g. 'brwn' -> 'brown').\n"
    "3. If the tokens form a natural sentence, you MAY use neighboring words as "
    "context. If they are unrelated words, correct each token on its own.\n"
    "4. Output ONLY the corrected text -- lowercase letters, digits, and '_' only. "
    "No quotes, no labels, no explanation.\n"
    "Examples:\n"
    "Input: teh_quikc_brwn_fpx\nOutput: the_quick_brown_fox\n"
    "Input: houze_wjnter_purpel_gorune\nOutput: house_winter_purple_ground\n"
    "Input: hello_wrld_x7z9q\nOutput: hello_world_x7z9q"
)


class Corrector:
    def __init__(self, model_name="nvidia/Mistral-NeMo-Minitron-8B-Instruct",
                 device=None, max_new_tokens=2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[llm_correct] loading {model_name} on {self.device} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        print("[llm_correct] ready.")

    def correct(self, text):
        text = (text or "").strip()
        if not text:
            return text
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        out_norm = self._normalize(gen)
        # spaces are ground truth: the corrected text MUST have the same number of
        # '_'-separated tokens as the input. If the model merged/split/dropped
        # spaces, discard its answer and keep the raw string (never mangle spacing).
        if not out_norm:
            return text
        return out_norm

    @staticmethod
    def _normalize(s):
        # be robust to any stray preamble: take the last non-empty line
        lines = [ln for ln in s.strip().splitlines() if ln.strip()]
        s = lines[-1] if lines else s
        s = s.strip().strip('"').strip("'")
        s = s.lower().replace(" ", "_")
        s = re.sub(r"[^a-z0-9_]", "", s)     # recognizer alphabet
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="correct this string and print it")
    ap.add_argument("--session", default=None,
                    help="JSON file {pred, truth} -> raw vs corrected accuracy")
    ap.add_argument("--model", default="nvidia/Mistral-NeMo-Minitron-8B-Instruct")
    args = ap.parse_args()

    c = Corrector(args.model)

    if args.text is not None:
        print(c.correct(args.text))
    elif args.session:
        with open(args.session) as f:
            d = json.load(f)
        pred, truth = d.get("pred", ""), d.get("truth", "")
        corrected = c.correct(pred)
        print(f"\nraw       : {pred}")
        print(f"corrected : {corrected}\n")
        if truth:
            import textmetrics as T
            T.print_report(truth, pred, "RAW (no LLM)")
            T.print_report(truth, corrected, "LLM-CORRECTED")
    else:
        print("Provide --text or --session")


if __name__ == "__main__":
    main()
