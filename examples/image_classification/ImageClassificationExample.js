class ImageClassificationExample extends BaseCameraExample {
  constructor(models) {
    super(models);
  }

  _customUI = () => {
    $('#fullscreen i svg').click(() => {
      $('video').toggleClass('fullscreen');
    });
  };

  _createRunner = () => {
    const runner = new ImageClassificationRunner();
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  _predict = async () => {
    const drawOptions = {
      inputSize: this._currentModelInfo.inputSize,
      preOptions: this._currentModelInfo.preOptions,
      imageChannels: 4,
      isDNNL:this._currentModelInfo.isDNNL,
    };
    await this._runner.run(this._currentInputElement, drawOptions);
    this._processOutput();
  };

  _processCustomOutput = () => {
    const output = this._runner.getOutput();
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    const labelClasses = getTopClasses(output.outputTensor, output.labels, 3, deQuantizeParams);
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  };
}
