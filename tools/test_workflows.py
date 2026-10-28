"""CPU checks for every bundled workflow: the shipped files match the builder, every node
matches its ComfyUI v0.35.0 schema, links are typed and complete, the Krea 2 settings,
and setup.sh downloads every model file a workflow loads.

    python tools/test_workflows.py

Core node schemas (socket inputs, outputs, number of widget values, widgets that may be
turned into inputs) are taken from ComfyUI v0.35.0 (nodes.py, comfy_extras/) and from
Comfy-Org's own Krea 2 workflows (github.com/Comfy-Org/workflow_templates,
image_krea2_turbo_*.json), which record how the frontend saves each node.
"""
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

# type: (socket inputs [(name, type, optional)], outputs [(name, type)], widget count,
#        widgets that may become inputs)
CORE = {
    "UNETLoader": ([], [("MODEL", "MODEL")], 2, ()),
    "CLIPLoader": ([], [("CLIP", "CLIP")], 3, ()),
    "VAELoader": ([], [("VAE", "VAE")], 1, ()),
    "LoraLoaderModelOnly": ([("model", "MODEL", False)], [("MODEL", "MODEL")], 2, ()),
    "CLIPTextEncode": ([("clip", "CLIP", False)], [("CONDITIONING", "CONDITIONING")], 1, ("text",)),
    "ConditioningZeroOut": ([("conditioning", "CONDITIONING", False)],
                            [("CONDITIONING", "CONDITIONING")], 0, ()),
    "EmptyLatentImage": ([], [("LATENT", "LATENT")], 3, ()),
    # seed + the frontend's control_after_generate + steps, cfg, sampler, scheduler, denoise
    "KSampler": ([("model", "MODEL", False), ("positive", "CONDITIONING", False),
                  ("negative", "CONDITIONING", False), ("latent_image", "LATENT", False)],
                 [("LATENT", "LATENT")], 7, ("seed",)),
    "VAEDecode": ([("samples", "LATENT", False), ("vae", "VAE", False)], [("IMAGE", "IMAGE")], 0, ()),
    "SaveImage": ([("images", "IMAGE", False)], [("images", "IMAGE")], 1, ()),
    "LoadImage": ([], [("IMAGE", "IMAGE"), ("MASK", "MASK")], 2, ()),
    "TextEncodeQwenImageEditPlus": ([("clip", "CLIP", False), ("vae", "VAE", True),
                                     ("image1", "IMAGE", True), ("image2", "IMAGE", True),
                                     ("image3", "IMAGE", True)],
                                    [("CONDITIONING", "CONDITIONING")], 1, ("prompt",)),
    "FluxKontextMultiReferenceLatentMethod": ([("conditioning", "CONDITIONING", False)],
                                              [("CONDITIONING", "CONDITIONING")], 1, ()),
    "ModelSamplingFlux": ([("model", "MODEL", False)], [("MODEL", "MODEL")], 4, ()),
    # is_input_list node; texts is a force_input STRING socket
    "SaveImageTextDataSetToFolder": ([("images", "IMAGE", False), ("texts", "STRING", True)], [], 3, ()),
    "MarkdownNote": ([], [], 1, ()),
    # training (comfy_extras/nodes_dataset.py, nodes_train.py)
    "MakeTrainingDataset": ([("images", "IMAGE", False), ("vae", "VAE", False), ("clip", "CLIP", False),
                             ("texts", "STRING", True)],
                            [("latents", "LATENT"), ("conditioning", "CONDITIONING")], 0, ()),
    # 18 settings: batch_size ... bypass_mode; its seed has no control_after_generate (the
    # frontend adds none for a plain seed: Comfy-Org's saved TextGenerate shows it)
    "TrainLoraNode": ([("model", "MODEL", False), ("latents", "LATENT", False),
                       ("positive", "CONDITIONING", False)],
                      [("lora", "LORA_MODEL"), ("loss_map", "LOSS_MAP"), ("steps", "INT")], 18, ()),
    "SaveLoRA": ([("lora", "LORA_MODEL", False)], [], 2, ("steps",)),
    "LossGraphNode": ([("loss", "LOSS_MAP", False)], [], 1, ()),
    "LoraModelLoader": ([("model", "MODEL", False), ("lora", "LORA_MODEL", False)],
                        [("model", "MODEL")], 2, ()),
}
TRAIN_ORDER = ["batch_size", "grad_accumulation_steps", "steps", "learning_rate", "rank", "optimizer",
               "loss_function", "seed", "training_dtype", "lora_dtype", "quantized_backward",
               "algorithm", "gradient_checkpointing", "checkpoint_depth", "offloading",
               "existing_lora", "bucket_mode", "bypass_mode"]
KREA_SAMPLER = [8, 1.0, "euler", "simple", 1.0]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load("builder", ROOT / "tools/build_workflow.py")
try:
    import torch  # noqa: F401  (the node pack imports it; the schema checks do not need it)
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")
pack = load("krea2_pack", ROOT / "custom_nodes/comfyui-krea2-batch/__init__.py")
SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")


def schema_of(ntype):
    """Core table, or the node pack's own INPUT_TYPES / RETURN_TYPES."""
    if ntype in CORE:
        return CORE[ntype]
    cls = pack.NODE_CLASS_MAPPINGS[ntype]
    req = cls.INPUT_TYPES()["required"]
    sockets = [(k, v[0], False) for k, v in req.items()
               if not (isinstance(v[0], list) or v[0] in ("INT", "FLOAT", "STRING", "BOOLEAN"))]
    widgets = [k for k, v in req.items() if (k, v[0], False) not in sockets]
    names = getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)
    return sockets, list(zip(names, cls.RETURN_TYPES)), len(widgets), tuple(widgets)


class WorkflowTests(unittest.TestCase):
    def each(self):
        for w in builder.WORKFLOWS:
            wf = builder.build(w["kind"])
            yield w, wf, {n["id"]: n for n in wf["nodes"]}

    def test_shipped_files_match_builder(self):
        shipped = {p.name for p in (ROOT / "workflows").glob("*.json")}
        self.assertEqual(shipped, {w["path"] for w in builder.WORKFLOWS})
        self.assertEqual(len({w["id"] for w in builder.WORKFLOWS}), len(builder.WORKFLOWS))
        for w, wf, _ in self.each():
            with self.subTest(workflow=w["path"]):
                path = ROOT / "workflows" / w["path"]
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), wf)

    def test_nodes_match_schemas(self):
        for w, wf, nodes in self.each():
            for n in nodes.values():
                with self.subTest(workflow=w["path"], node=n["id"], type=n["type"]):
                    sockets, outputs, widget_count, convertible = schema_of(n["type"])
                    self.assertEqual([(o["name"], o["type"]) for o in n["outputs"]], outputs)
                    socket_inputs = [i for i in n["inputs"] if "widget" not in i]
                    self.assertEqual([(i["name"], i["type"]) for i in socket_inputs],
                                     [(s, t) for s, t, _ in sockets])
                    for i, (_, _, optional) in zip(socket_inputs, sockets):
                        self.assertEqual(i.get("shape") == 7, optional, i["name"])
                    for i in n["inputs"]:
                        if "widget" in i:
                            self.assertIn(i["widget"]["name"], convertible)
                    self.assertEqual(len(n.get("widgets_values", [])), widget_count)

    def test_links_are_typed_and_required_inputs_connected(self):
        for w, wf, nodes in self.each():
            with self.subTest(workflow=w["path"]):
                ids = set()
                for lid, src, slot, dst, port, kind in wf["links"]:
                    ids.add(lid)
                    out, inp = nodes[src]["outputs"][slot], nodes[dst]["inputs"][port]
                    self.assertEqual((out["type"], inp["type"]), (kind, kind))
                    self.assertEqual(inp["link"], lid)
                    self.assertIn(lid, out["links"])
                self.assertEqual(len(ids), len(wf["links"]))
                for n in nodes.values():
                    for i in n["inputs"]:
                        if i.get("shape") != 7:
                            self.assertIsNotNone(i["link"], f"{n['type']}.{i['name']} not connected")
                self.assertEqual(wf["last_link_id"], max(ids))
                self.assertEqual(wf["last_node_id"], max(nodes))

    def test_krea2_settings(self):
        for w, wf, nodes in self.each():
            with self.subTest(workflow=w["path"]):
                by_type = {}
                for n in nodes.values():
                    by_type.setdefault(n["type"], []).append(n)
                (ks,) = by_type["KSampler"]
                self.assertEqual(ks["widgets_values"][2:], KREA_SAMPLER)
                (clip,) = by_type["CLIPLoader"]
                self.assertEqual(clip["widgets_values"][1], "krea2")
                # CFG 1: the negative is the zeroed positive, as in Comfy-Org's templates
                neg = next(i for i in ks["inputs"] if i["name"] == "negative")
                src = next(l for l in wf["links"] if l[0] == neg["link"])[1]
                self.assertEqual(nodes[src]["type"], "ConditioningZeroOut")

    def test_list_runs_get_one_seed_per_prompt(self):
        for w, wf, nodes in self.each():
            if w["kind"] == "single":
                continue
            with self.subTest(workflow=w["path"]):
                (ks,) = [n for n in nodes.values() if n["type"] == "KSampler"]
                seed = next(i for i in ks["inputs"] if i["name"] == "seed")
                src, slot = next((l[1], l[2]) for l in wf["links"] if l[0] == seed["link"])
                self.assertEqual((nodes[src]["type"], nodes[src]["outputs"][slot]["name"]),
                                 ("Krea2PromptList", "seeds"))

    def test_reference_workflows_use_the_portrait(self):
        for w, wf, nodes in self.each():
            if w["kind"] not in ("views", "dataset"):
                continue
            with self.subTest(workflow=w["path"]):
                (enc,) = [n for n in nodes.values() if n["type"] == "TextEncodeQwenImageEditPlus"]
                linked = {i["name"] for i in enc["inputs"] if i["link"] is not None}
                self.assertTrue({"clip", "vae", "image1", "prompt"} <= linked)
                pl = next(n for n in nodes.values() if n["type"] == "Krea2PromptList")
                lines = pack.read_lines(pl["widgets_values"][0], "")
                self.assertEqual(len(lines), 8 if w["kind"] == "views" else 24)
        (ds,) = [n for _, wf, nodes in self.each() for n in nodes.values()
                 if n["type"] == "SaveImageTextDataSetToFolder"]
        self.assertIsNotNone(next(i for i in ds["inputs"] if i["name"] == "texts")["link"])

    def test_training_workflow(self):
        wf = builder.build("train")
        nodes = {n["id"]: n for n in wf["nodes"]}
        src = {(l[3], nodes[l[3]]["inputs"][l[4]]["name"]): (l[1], nodes[l[1]]["outputs"][l[2]]["name"])
               for l in wf["links"]}
        (train,) = [n for n in nodes.values() if n["type"] == "TrainLoraNode"]
        settings = dict(zip(TRAIN_ORDER, train["widgets_values"]))
        self.assertEqual(settings, builder.TRAIN)
        self.assertTrue(settings["bypass_mode"])          # FP8 weights stay frozen
        self.assertIn(settings["algorithm"], ("LoRA", "LoHa", "LoKr", "OFT"))
        self.assertIn(settings["optimizer"], ("AdamW", "Adam", "SGD", "RMSprop"))
        self.assertIn(settings["training_dtype"], ("bf16", "fp32", "none"))
        # trained on Raw, tested on Turbo
        model_src = nodes[src[(train["id"], "model")][0]]
        self.assertEqual(model_src["widgets_values"][0], builder.RAW_UNET)
        (apply,) = [n for n in nodes.values() if n["type"] == "LoraModelLoader"]
        self.assertEqual(nodes[src[(apply["id"], "model")][0]]["widgets_values"][0], builder.UNET)
        self.assertEqual(src[(apply["id"], "lora")], (train["id"], "lora"))
        (ks,) = [n for n in nodes.values() if n["type"] == "KSampler"]
        self.assertEqual(src[(ks["id"], "model")], (apply["id"], "model"))
        # the dataset is encoded with its captions, and the LoRA is saved with its step count
        (make,) = [n for n in nodes.values() if n["type"] == "MakeTrainingDataset"]
        self.assertEqual(nodes[src[(make["id"], "texts")][0]]["type"], "Krea2LoadDataset")
        (save,) = [n for n in nodes.values() if n["type"] == "SaveLoRA"]
        self.assertEqual(src[(save["id"], "steps")], (train["id"], "steps"))
        self.assertTrue(save["widgets_values"][0].startswith("loras/"))   # output/loras is registered
        # it reads the folder the training-set workflow writes
        (load,) = [n for n in nodes.values() if n["type"] == "Krea2LoadDataset"]
        (ds,) = [n for n in builder.build("dataset")["nodes"] if n["type"] == "SaveImageTextDataSetToFolder"]
        self.assertEqual(load["widgets_values"], [ds["widgets_values"][0], "output"])
        trigger = builder.TRIGGER.rstrip(",")
        tests = next(n for n in nodes.values() if n["type"] == "Krea2PromptList")
        self.assertTrue(all(l.startswith(trigger) for l in pack.read_lines(tests["widgets_values"][0], "")))

    def test_every_workflow_states_its_gpu_needs(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        gpu = readme[readme.index("\n## GPU\n"):]
        gpu = gpu[:gpu.index("\n## ", 5)]
        for n, (w, wf, nodes) in enumerate(self.each(), 1):
            with self.subTest(workflow=w["path"]):
                start = next(x for x in nodes.values() if x.get("title") == "Start here")
                text = start["widgets_values"][0]
                self.assertRegex(text, r"\*\*GPU[^*]*:\*\* \d+ GB minimum")
                self.assertIn("**Out of memory", text)
                self.assertRegex(gpu, rf"\n\| {n} [A-Z]")       # a row in the README's GPU table
        self.assertIn("48 GB", builder.GPU_NOTES["train"])
        self.assertIn("not yet measured", builder.GPU_NOTES["train"])

    def test_setup_downloads_every_model_a_workflow_loads(self):
        fetched = set(re.findall(r'download_model "([a-z_]+/[^"]+)"', SETUP))
        fetched |= {f"loras/krea2_{n}.safetensors" for n in re.search(
            r"for lora in ([^;]+);", SETUP).group(1).replace("\\\n", " ").split()}
        folders = {"UNETLoader": "diffusion_models", "CLIPLoader": "text_encoders",
                   "VAELoader": "vae", "LoraLoaderModelOnly": "loras"}
        for w, wf, nodes in self.each():
            for n in nodes.values():
                if n["type"] in folders:
                    with self.subTest(workflow=w["path"], node=n["type"]):
                        self.assertIn(f"{folders[n['type']]}/{n['widgets_values'][0]}", fetched)
        self.assertIn("diffusion_models/krea2_turbo_fp8_scaled.safetensors", fetched)

    def test_inputs_the_workflows_open_on_ship_with_the_repo(self):
        for w, wf, nodes in self.each():
            for n in nodes.values():
                if n["type"] == "LoadImage":
                    self.assertTrue((ROOT / "samples" / n["widgets_values"][0]).is_file())
        self.assertTrue((ROOT / "prompts" / builder.BATCH_FILE).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=1)
