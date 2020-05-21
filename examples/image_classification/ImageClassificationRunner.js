class ImageClassificationRunner extends BaseRunner {
  constructor() {
    super();
    this._labels = null;
  }

  _setLabels = (labels) => {
    this._labels = labels;
  };

  _getLabels = async (url) => {
    const result = await this._loadURL(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

  _getInputTensorTypedArray = () => {
    let typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
    if (this._currentModelInfo.isDNNL) typedArray = Int8Array;
    return typedArray;
  };

  _getOutputTensorTypedArray = () => {
    let typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
    if (this._currentModelInfo.isDNNL) typedArray = Int8Array;
    return typedArray;
  };

  _getOtherResources = async () => {
    await this._getLabels(this._currentModelInfo.labelsFile);
  };

  _updateOutput = (output) => {
    output.labels = this._labels;
  };
}
