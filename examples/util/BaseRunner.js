class BaseRunner {
  constructor() {
    this._currentBackend = null;
    this._currentPrefer = null;
    this._currentModelInfo = {};
    this._inputTensor = [];
    this._outputTensor = [];
    this._currentRequest = null;
    this._progressHandler = null;
    this._rawModel = null;
    this._bLoaded = false; // loaded status of raw model for Web NN API
    this._model = null; // get Web NN model by converting raw model
    this._subgraphsSummary = [];
    this._modelRequiredOps = null;
    this._deQuantizeParams = null;
    this._bInitialized = false; // initialized status for model
    this._bEagerMode = false;
    this._supportedOps = new Set();
    this._inferenceTime = 0.0; // ms
  }

  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo
  };

  _setRequest = (req) => {
    // Record current request, aborts the request if it has already been sent
    this._currentRequest = req;
  };

  setProgressHandler = (handler) => {
    // Use handler to prompt for model loading progress info
    this._progressHandler = handler;
  };

  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  _setRawModel = (rawModel) => {
    this._rawModel = rawModel;
  };

  _setLoadedFlag = (flag) => {
    this._bLoaded = flag;
  };

  _setModel = (model) => {
    this._model = model;
  };

  _setSubgraphsSummary = (summary) => {
    this._subgraphsSummary = summary;
  };

  _setModelRequiredOps = (ops) => {
    this._modelRequiredOps = ops;
  };

  _setDeQuantizeParams = (params) => {
    this._deQuantizeParams = params;
  };

  _setInitializedFlag = (flag) => {
    this._bInitialized = flag;
  };

  _setInferenceTime = (t) => {
    this._inferenceTime = t;
  };

  _loadURL = async (url, handle = null, isBinary = false) => {
    let _this = this;
    return new Promise((resolve, reject) => {
      if (_this._currentRequest != null) {
        _this._currentRequest.abort();
      }
      let oReq = new XMLHttpRequest();
      _this._setRequest(oReq);
      oReq.open('GET', url, true);
      if (isBinary) {
        oReq.responseType = 'arraybuffer';
      }
      oReq.onload = function (ev) {
        _this._setRequest(null);
        if (oReq.readyState === 4) {
          if (oReq.status === 200) {
            resolve(oReq.response);
          } else {
            reject(new Error(`Failed to load ${url} . Status: [${oReq.status}]`));
          }
        }
      };
      if (handle != null) {
        oReq.onprogress = handle;
      }
      oReq.send();
    });
  };

  _getRawModel = async (url) => {
    let status = 'ERROR';
    let rawModel = null;

    if (url !== undefined) {
      const arrayBuffer = await this._loadURL(url, this._progressHandler, true);
      const bytes = new Uint8Array(arrayBuffer);
      switch (url.split('.').pop()) {
        case 'tflite':
          const flatBuffer = new flatbuffers.ByteBuffer(bytes);
          rawModel = tflite.Model.getRootAsModel(flatBuffer);
          rawModel._rawFormat = 'TFLITE';
          status = 'SUCCESS'
          printTfLiteModel(rawModel);
          break;
        case 'onnx':
          const err = onnx.ModelProto.verify(bytes);
          if (err) {
            throw new Error(`The model file ${url} is invalid, ${err}`);
          }
          rawModel = onnx.ModelProto.decode(bytes);
          rawModel._rawFormat = 'ONNX';
          status = 'SUCCESS'
          printOnnxModel(rawModel);
          break;
        case 'bin':
          const networkFile = url.replace(/bin$/, 'xml');
          const networkText = await this._loadURL(networkFile);
          const weightsBuffer = bytes.buffer;
          rawModel = new OpenVINOModel(networkText, weightsBuffer);
          rawModel._rawFormat = 'OPENVINO';
          status = 'SUCCESS';
          break;
          case 'pb':
            const weightFile = url.replace(/predict/, 'init');
            const weightBuffer = await this._loadURL(weightFile, this._progressHandler, true);
            const weightBytes = new Uint8Array(weightBuffer);
            const netBuffer = bytes;
            const weightMessage = protobuf.roots["caffe2"].caffe2.NetDef.decode(weightBytes);
            const netMessage = protobuf.roots["caffe2"].caffe2.NetDef.decode(netBuffer);
            const caffe2Utils = new Caffe2ModelUtils(netMessage,
                                                     weightMessage,
                                                     this._currentModelInfo.isQuantized,
                                                     this._currentModelInfo.isDNNL);
            rawModel = [...caffe2Utils.getCaffe2Model()];
            rawModel._rawFormat = 'CAFFE2';
            status = 'SUCCESS';
            break;
        default:
          throw new Error(`Unrecognized model format, support TFLite | ONNX | OpenVINO model`);
      }
    } else {
      throw new Error(`There's none model file info, please check config info of modelZoo.`);
    }

    this._setRawModel(rawModel);
    this._setLoadedFlag(true);
    return status;
  };

  _getOtherResources = async () => {
    // Override by inherited if needed, likes load labels file
  };

  _getModelResources = async () => {
    await this._getRawModel(this._currentModelInfo.modelFile);
    await this._getOtherResources();
  };

  loadModel = async (modelInfo) => {
    if (this._bLoaded && this._currentModelInfo.modelFile === modelInfo.modelFile) {
      return 'LOADED';
    }

    // reset all states
    this._setLoadedFlag(false);
    this._setInitializedFlag(false);
    this._setBackend(null);
    this._setPrefer(null);
    this._setModelInfo(modelInfo);
    this._setModelRequiredOps(new Set());
    this._setDeQuantizeParams([]);
    this._setSubgraphsSummary([]);
    this._initInputTensor();
    this._initOutputTensor();

    await this._getModelResources();
  };

  _getInputTensorTypedArray = () => {
    // Override by inherited if needed
    const typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
    return typedArray;
  };

  _initInputTensor = () => {
    const typedArray = this._getInputTensorTypedArray();
    this._inputTensor = [new typedArray(this._currentModelInfo.inputSize.reduce((a, b) => a * b))];
  };

  _getOutputTensorTypedArray = () => {
    // Override by inherited if needed
    const typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
    return typedArray;
  };

  _initOutputTensor = () => {
    // Override by inherited if needed
    const typedArray = this._getOutputTensorTypedArray();
    const outputSize = this._currentModelInfo.outputSize;

    if (typeof outputSize === 'number') {
      this._outputTensor = [new typedArray(outputSize)];
    } else {
      this._outputTensor = [new typedArray(outputSize.reduce((a, b) => a * b))];
    }
  };

  compileModel = async (backend, prefer) => {
    if (!this._bLoaded) {
      return 'NOT_LOADED';
    }

    if (this._bInitialized && backend === this._currentBackend && prefer === this._currentPrefer) {
      return 'INITIALIZED';
    }

    this._setBackend(backend);
    this._setPrefer(prefer);
    this._setInitializedFlag(false);
    const postOptions = this._currentModelInfo.postOptions || {};
    const configs = {
      rawModel: this._rawModel,
      backend: this._currentBackend,
      prefer: this._currentPrefer,
      softmax: postOptions.softmax || false,
      isQuantized: this._currentModelInfo.isQuantized || false,
      isDNNL: this._currentModelInfo.isDNNL || false,
      inputSize: this._currentModelInfo.inputSize || [224, 224, 3]
    };

    let model = null;

    switch (this._rawModel._rawFormat) {
      case 'TFLITE':
        model = new TFliteModelImporter(configs);
        break;
      case 'ONNX':
        model = new OnnxModelImporter(configs);
        break;
      case 'OPENVINO':
        model = new OpenVINOModelImporter(configs);
        break;
      case 'CAFFE2':
        model = new Caffe2ModelImporter(configs);
        break;
      default:
        throw new Error(`Unsupported '${rawModel._rawFormat}' input.`);
    }

    this._setModel(model);
    this._model.setSupportedOps(this._supportedOps);
    this._model.setEagerMode(this._bEagerMode);
    const compileStatus = await this._model.createCompiledModel();
    console.log(`Compilation Status: [${compileStatus}]`);

    this._setModelRequiredOps(this._model.getRequiredOps());

    if (this._currentModelInfo.isQuantized) {
      this._setDeQuantizeParams(model._deQuantizeParams);
    }

    if (this._currentBackend !== 'WebML' && model._compilation && model._compilation._preparedModel) {
       this._setSubgraphsSummary(model._compilation._preparedModel.getSubgraphsSummary());
    }

    // Warm up model
    const computeStart = performance.now();
    const computeStatus = await this._model.compute(this._inputTensor, this._outputTensor);
    const computeDelta = performance.now() - computeStart;
    console.log(`Computed Status: [${computeStatus}]`);
    console.log(`Warm up Time: ${computeDelta.toFixed(2)} ms`);

    this._setInitializedFlag(true);
    return 'SUCCESS';
  };

  run = async (src, options) => {
    let status = 'ERROR';

    if (src.tagName === 'AUDIO') {
      await getTensorArrayByAudio(src, this._inputTensor, options);
    } else {
      getTensorArray(src, this._inputTensor, options);
    }
    console.log(this._inputTensor);  // delete
    this._initOutputTensor();  // delete

    const start = performance.now();
    status = await this._model.compute(this._inputTensor, this._outputTensor);

    console.log(this._outputTensor);  // delete

    if (this._currentModelInfo.isDNNL) {
      let output = Array.from(this._outputTensor);
      let outputTmp = [];
      let expSum = 0;
      let maxNum = 0;

      for (let i = 0; i < output[0].length; i++) {
        maxNum = Math.max(maxNum, output[0][i]);
      }
      console.log(maxNum);

      for (let i = 0; i < output[0].length; i++) {
        expSum = expSum + Math.exp((output[0][i] - maxNum));
      }
      console.log(expSum);

      for (let i = 0; i < output[0].length; i++) {
        let tmpNum = (Math.exp(output[0][i] - maxNum) / expSum).toFixed(4);
        outputTmp.push(tmpNum);
      }

      console.log(output);
      this._outputTensor = [new Float32Array(outputTmp)];
    }
    console.log(this._outputTensor);  // delete


    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    console.log(`Computed Status: [${status}]`);
    console.log(`Compute Time: [${delta} ms]`);
    return status;
  };

  getRequiredOps = () => {
    return this._modelRequiredOps;
  };

  getSubgraphsSummary = () => {
    return this._subgraphsSummary;
  };

  getDeQuantizeParams = () => {
    return this._deQuantizeParams;
  };

  _updateOutput = (output) => {
    // Override by inherited if needed
  };

  getOutput = () => {
    let output = {
      outputTensor: this._outputTensor[0],
      inferenceTime: this._inferenceTime,
    };
    this._updateOutput(output); // add custom output info
    return output;
  };

  deleteAll = () => {
    if (this._currentBackend != 'WebML') {
      // free allocated memory on compilation process by polyfill WASM / WebGL backend.
      if (this._model._compilation && this._model._compilation._preparedModel) {
        this._model._compilation._preparedModel._deleteAll();
      }
    }
  };

  // for debugging
  iterateLayers = async (configs, layerList) => {
    if (!this._bInitialized) return;

    const iterators = [];
    const models = [];

    for (const config of configs) {
      const fileExtension = this._currentModelInfo.modelFile.split('.').pop();
      const importer = {
        tflite: TFliteModelImporter,
        onnx: OnnxModelImporter,
        bin: OpenVINOModelImporter,
      }[fileExtension];
      const model = await new importer({
        isQuantized: this._currentModelInfo.isQuantized,
        rawModel: this._rawModel,
        backend: config.backend,
        prefer: config.prefer || null,
      });
      iterators.push(model.layerIterator(this._inputTensor, layerList));
      models.push(model);
    }

    while (true) {
      let layerOutputs = [];
      for (let it of iterators) {
        layerOutputs.push(await it.next());
      }
      let refOutput = layerOutputs[0];
      if (refOutput.done) {
        break;
      }
      console.debug(`\n\n\nLayer(${refOutput.value.layerId}) ${refOutput.value.outputName}`);
      for (let i = 0; i < configs.length; ++i) {
        console.debug(`\n${configs[i].backend}:`);
        console.debug(`\n${layerOutputs[i].value.tensor}`);
        if (i > 0) {
          let sum = 0;
          for (let j = 0; j < refOutput.value.tensor.length; j++) {
            sum += Math.pow(layerOutputs[i].value.tensor[j] - refOutput.value.tensor[j], 2);
          }
          let variance = sum / refOutput.value.tensor.length;
          console.debug(`var with ${configs[0].backend}: ${variance}`);
        }
      }
    }

    for (let model of models) {
      if (model._backend !== 'WebML') {
        model._compilation._preparedModel._deleteAll();
      }
    }
  };
}
