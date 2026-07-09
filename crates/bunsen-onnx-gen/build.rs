use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("onnx/silero_vad_op18_ifless.onnx")
        .out_dir("./")
        .run_from_script();
}
