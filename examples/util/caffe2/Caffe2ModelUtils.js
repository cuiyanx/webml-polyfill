class Caffe2ModelUtils {
  constructor(predictModel, initModel, isQuantized=false, isDNNL=false) {
    this._predict = predictModel;
    this._init = initModel;
    this._quantized = isQuantized;
    this._isDNNL = isDNNL;
    this._predictDataFormat = false;    // NCHW to NHWC
    this._initDataFormat = false;       // NCHW to NHWC

    this._checkDataFormat();
    this._initMap = this._initModelHandler();
    this._predictMap = this._predictModelHandler();
  }

  getCaffe2Model () {
    return this._predictMap;
  }

  getCaffe2InitModel () {
    return this._initMap;
  }

  _initModelHandler () {
    let initTensorMap = [];

    for (let opIdx in this._init.op) {
      initTensorMap[opIdx] = [];
      let op = this._init.op[opIdx];
      initTensorMap[opIdx][op.output] = [];

      for (let argIdx in op.arg) {
        let arg = op.arg[argIdx];
        initTensorMap[opIdx][op.output][arg.name] = [];
        let data = this._checkArgData(arg);
        initTensorMap[opIdx][op.output][arg.name]["type"] = data.type;
        initTensorMap[opIdx][op.output][arg.name]["value"] = data.value;
      }

      // uint8 => int8
      if (this._quantized && this._isDNNL &&
          typeof initTensorMap[opIdx][op.output]["values"] != "undefined" &&
          initTensorMap[opIdx][op.output]["values"]["type"] == "uint8" &&
          typeof initTensorMap[opIdx][op.output]["Y_zero_point"] != "undefined" &&
          initTensorMap[opIdx][op.output]["Y_zero_point"]["value"] == "128") {
        initTensorMap[opIdx][op.output]["values"]["type"] = "int8";
        let tmpArray = [];
        for ( let val of Object.values(initTensorMap[opIdx][op.output]["values"]["value"])) {
          tmpArray.push(val - 128);
        }
        initTensorMap[opIdx][op.output]["values"]["value"] = tmpArray;
        initTensorMap[opIdx][op.output]["Y_zero_point"]["value"] = "0";
      }

      // Data handle
      if (typeof initTensorMap[opIdx][op.output]["shape"] != "undefined" &&
          typeof initTensorMap[opIdx][op.output]["values"] != "undefined") {
        initTensorMap[opIdx][op.output] = this._DataHandleforTensor(initTensorMap[opIdx][op.output]);
      }
    }

    return initTensorMap;
  }

  _predictModelHandler() {
    let predictTensorMap = [];

    for (let opIdx in this._predict.op) {
      predictTensorMap[opIdx] = [];
      let op = this._predict.op[opIdx];

      // name, type, engine
      predictTensorMap[opIdx]["name"] = op.name;
      predictTensorMap[opIdx]["operator"] = op.type;
      predictTensorMap[opIdx]["engine"] = op.engine;

      // input
      predictTensorMap[opIdx]["input"] = [];
      for (let inputIdx in op.input) {
        predictTensorMap[opIdx]["input"][inputIdx] = [];
        let input = op.input[inputIdx];
        predictTensorMap[opIdx]["input"][inputIdx]["name"] = input;

        for (let inTmp of this._initMap) {
          if (inTmp.hasOwnProperty(input)) {
            for (let key in inTmp[input]) {
              predictTensorMap[opIdx]["input"][inputIdx][key] = inTmp[input][key];
            }
          }
        }
      }

      // output
      predictTensorMap[opIdx]["output"] = [];
      for (let outputIdx in op.output) {
        predictTensorMap[opIdx]["output"][outputIdx] = [];
        let output = op.output[outputIdx];
        predictTensorMap[opIdx]["output"][outputIdx]["name"] = output;
      }

      // arg
      predictTensorMap[opIdx]["arg"] = [];
      for (let argIdx in op.arg) {
        let arg = op.arg[argIdx];

        predictTensorMap[opIdx]["arg"][arg.name] = [];
        let data = this._checkArgData(arg);
        predictTensorMap[opIdx]["arg"][arg.name]["type"] = data.type;
        predictTensorMap[opIdx]["arg"][arg.name]["value"] = data.value;

        if (arg.name == "order") {
          predictTensorMap[opIdx]["arg"][arg.name]["type"] = "str";
          let orderTmp = [];
          for (let val of predictTensorMap[opIdx]["arg"][arg.name]["value"]) {
            orderTmp.push(String.fromCharCode(val));
          }
          predictTensorMap[opIdx]["arg"][arg.name]["value"] = orderTmp.join("");
        }

        // Data handle
        if (arg.name != "order" &&
            typeof predictTensorMap[opIdx]["arg"][arg.name]["type"] != "undefined" &&
            typeof predictTensorMap[opIdx]["arg"][arg.name]["value"] != "undefined") {
          predictTensorMap[opIdx]["arg"][arg.name] = this._DataHandleforArg(predictTensorMap[opIdx]["arg"][arg.name]);
        }
      }
    }

    return predictTensorMap;
  }

  _checkDataFormat() {
    this._checkInitDataFormat();
    this._checkpredictDataFormat();
    // console.log("initDataFormat: " + this._initDataFormat);
    // console.log("_predictDataFormat: " + this._predictDataFormat);
  }

  _checkInitDataFormat() {
    for (let op of this._init.op) {
      for (let arg of op.arg) {
        if (arg.name == "order") {
          let data = this._checkArgData(arg);
          let orderTmp = [];
          for (let val of data.value) {
            orderTmp.push(String.fromCharCode(val));
          }
          let formatStr = orderTmp.join("");
          if (formatStr == "NCHW") {
            this._initDataFormat = true;
            return
          };
        }
      }
    }
  }

  _checkpredictDataFormat() {
    for (let op of this._predict.op) {
      for (let arg of op.arg) {
        if (arg.name == "order") {
          let data = this._checkArgData(arg);
          let orderTmp = [];
          for (let val of data.value) {
            orderTmp.push(String.fromCharCode(val));
          }
          let formatStr = orderTmp.join("");
          if (formatStr == "NCHW") {
            this._predictDataFormat = true;
            return
          };
        }
      }
    }
  }

  _checkArgData(arg) {
    for (let [key, val] of Object.entries(arg)) {
      if (key != "name" && key != "tensors" && key != "nets" && key != "qtensors") {
        if (val.length !== 0) {
          return this._pareData(val, key);
        }
      }
    }
  }

  _pareData(dataValue, dataType) {
    switch(dataType) {
      case "i":
      case "ints": {
        dataType = "int32";
      } break;
      case "f":
      case "floats": {
        dataType = "float32";
      } break;
      case "s": {
        dataType = "uint8";
        let dataTmp = [];
        let buf = new Uint8Array(dataValue);
        for (let value of buf.values()) {
          dataTmp.push(value);
        }
        dataValue = dataTmp;
      } break;
      default: {
        throw new Error(`${dataType} is not supported.`);
      }
    };

    return {"type": dataType, "value": dataValue};
  }

  _DataHandleforArg (arg) {
    let dataShape = arg["value"];
    let N = dataShape[0];
    let C = dataShape[1];
    let H = dataShape[2];
    let W = dataShape[3];

    if (this._predictDataFormat && dataShape.length === 4) {
      arg["value"] = [N, H, W, C];
    }

    return arg;
  }

  _DataHandleforTensor (tensor) {
    // For shape
    let dataShape = tensor["shape"]["value"];
    let N = dataShape[0];
    let C = dataShape[1];
    let H = dataShape[2];
    let W = dataShape[3];
    let flag = dataShape.length === 4 ? true : false;

    if (this._initDataFormat && flag) {
      tensor["shape"]["value"] = [N, H, W, C];
    }

    // For value
    let dataValue = tensor["values"]["value"];
    let typeValue = tensor["values"]["type"];
    let ctorValue = this._TypetoArray(typeValue);

    let tmpDataValue;
    if (this._initDataFormat && flag) {
      tmpDataValue = new ctorValue(dataValue.length);
      for (let n = 0; n < N; ++n) {
        for (let c = 0; c < C; ++c) {
          for (let h = 0; h < H; ++h) {
            for (let w = 0; w < W; ++w) {
              tmpDataValue[n*H*W*C + h*W*C + w*C + c] = dataValue[n*C*H*W + c*H*W + h*W + w];
            }
          }
        }
      }
    } else {
      tmpDataValue = new ctorValue(this._DatatoArray(dataValue));
    }

    tensor["values"]["value"] = tmpDataValue;

    return tensor;
  }

  _TypetoArray (type) {
    let ctor;
    if (type == "int32") ctor = Int32Array;
    else if (type == "uint32") ctor = Uint32Array;
    else if (type == "float32") ctor = Float32Array;
    else if (type == "uint8") ctor = Uint8Array;
    else if (type == "int8") ctor = Int8Array;
    else if (type == "str") ctor = Array;
    else throw new Error(`${type} is not supported.`);
    return ctor;
  }

  _DatatoArray (data) {
    let dataTmp = [];
    if (typeof data.length == "undefined" || data.length == 1) {
      dataTmp.push(data);
    } else {
      dataTmp = data;
    }
    return dataTmp;
  }
}
