from metaflow import FlowSpec, project, step, Parameter

from pipelines.common import DatasetMixin


@project(name="fusemycell")
class TestFlowMixinFlow(FlowSpec, DatasetMixin):
    """Pipeline used to test the FlowMixin class."""

    dataset_dir = Parameter(
        "dataset-dir",
        help="Directory containing the light sheet microscopy dataset.",
        default="data",  # Point to your data directory
    )
    
    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=4,
    )

    @step
    def start(self):  # noqa: D102
        self.data = self.load_dataset()
        self.next(self.end)

    @step
    def end(self):  # noqa: D102
        pass


if __name__ == "__main__":
    TestFlowMixinFlow()