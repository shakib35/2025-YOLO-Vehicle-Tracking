import os
from ultralytics import YOLO
import comet_ml
from comet_ml import start
import torch
import time

if __name__ == '__main__':
    # Comet environment variables
    os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "5"
    os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "false"

    COMET_SUPPORTED_TASKS = ["detect"]
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve", "confusion_matrix"
    LABEL_PLOT_NAMES = "labels", "labels_correlogram"

    # Comet logging setup
    experiment = start(
        api_key="W2y4B8fSkwtOQPElr8nni3A1H", #input api key
        project_name="2025 YOLOv11 Detection and Tracking",
        workspace="shakib35"
    )
    experiment_name = "yolo11s"
    experiment.set_name(experiment_name)

    # Parameters
    device = "cuda"
    model = YOLO("yolo11s.pt")

    # Logging Parameters to Comet
    experiment.log_parameter("device", device)
    experiment.log_parameter("epochs", 50)

    start_time = time.time()

    # Train model based on YAML configuration
    results = model.train(
        data="config.yaml",
        project="Car Tracking",
        batch=16,
        epochs=50,
        device=device
    )

    # Print runtime
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time} seconds")
    # experiment.log_metric(name="training_time", value=training_time)

    # Save final trained model
    model_path = 'yolov11s trained'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    experiment.log_model("trained_model", model_path)

    # Evaluate on validation data to calculate metrics
    metrics = model.val()

    #End the experiment
    experiment.end()
