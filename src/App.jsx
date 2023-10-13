import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // 设置后端为WebGL
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [models, setModels] = useState({ agemodel: null, genmodel: null, racemodel: null });

  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const modelNames = ["agemodel", "genmodel", "racemodel"];

  useEffect(() => {
    tf.ready().then(async () => {
      const loadedModels = {};
      for (let modelName of modelNames) {
        const model = await tf.loadLayersModel(`${window.location.href}/${modelName}_js/model.json`);
		console.log('step 1');

        loadedModels[modelName] = model;

        // 预热模型
        //const dummyInput = tf.ones(model.inputs[0].shape);
		const inputShape = model.inputs[0].shape;  // 假设 inputShape 为 [null, 200, 200, 3]
		const newShape = inputShape.slice(1);
		console.log('model.inputs[0].shape' + inputShape);
		const dummyInput = tf.ones(newShape);
		console.log('step 2');
        const warmupResults = model.predict(dummyInput.expandDims(0));
		console.log('step 3');
        tf.dispose([warmupResults, dummyInput]);
		console.log('step 4');
      }
      setModels(loadedModels);
      setLoading({ loading: false, progress: 1 });
    });
  }, []);

  return (
    <div className="App">
      {loading.loading && <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>}
      <div className="header">
        <h1>🤖 Gesichtserkennungsbasierte Vorhersage <br /> von Alter, Geschlecht und Rasse</h1>
        <p>Face Recognition-based Prediction of Age, Gender, and Race</p>
      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => detect(imageRef.current, models.agemodel,models.genmodel, models.racemodel,canvasRef.current)} // 使用正确的模型对象
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() => detectVideo(cameraRef.current, models.agemodel,models.genmodel, models.racemodel,canvasRef.current)} // 使用正确的模型对象
        />
        <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() => detectVideo(videoRef.current, models.agemodel,models.genmodel, models.racemodel, canvasRef.current)} // 使用正确的模型对象
        />
        <canvas width={models.agemodel?.inputs[0].shape[1]} height={models.agemodel?.inputs[0].shape[2]} ref={canvasRef} /> {/* 使用正确的模型对象和输入尺寸 */}
      </div>

      <ButtonHandler imageRef={imageRef} cameraRef={cameraRef} videoRef={videoRef} />
    </div>
  );
};

export default App;
