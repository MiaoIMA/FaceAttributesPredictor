import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";
//import labels from "./labels.json";

//const numClass = labels.length;

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor
 */
const preprocess = (source, modelWidth, modelHeight) => {
  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // Ensure the image is resized to the model expected size
    const resized = tf.image.resizeBilinear(img, [modelWidth, modelHeight]);

    // Normalize the image (pixel values from [0, 255] to [0, 1])
    const normalized = resized.div(255.0);

    // Add a batch dimension to the image so it can be fed into the model
    return normalized.expandDims(0);
  });

  return input;
};
;

/**
 * Funktion zur Durchführung von Inferenzen mit den drei Modellen und zur Anzeige der Ergebnisse.
 * @param {HTMLImageElement|HTMLVideoElement} source - Das Eingabebild oder -video-Element
 * @param {tf.LayersModel} agemodel - Das geladene Altersvorhersagemodell
 * @param {tf.LayersModel} genmodel - Das geladene Geschlechtsvorhersagemodell
 * @param {tf.LayersModel} racemodel - Das geladene Rassenvorhersagemodell
 * @param {HTMLCanvasElement} canvasRef - Der Canvas-Referenz
 * @param {VoidFunction} [callback = () => {}] - Callback-Funktion, die nach dem Erkennungsprozess ausgeführt wird
 */

export const detect = async (source, agemodel, genmodel, racemodel, canvasRef, callback = () => {}) => {
  const modelWidth = 200;
  const modelHeight = 200;

  const input = preprocess(source, modelWidth, modelHeight); 

  console.log(genmodel);

  const age = agemodel.predict(input);
  const genderProb = genmodel.predict(input);
  const raceProb = racemodel.predict(input);
  
  const agePred = age.dataSync()[0];
  console.log('age.dataSync ' + age.dataSync()[0]);
  const genderPred = genderProb.dataSync()[0] > 0.5 ? 'Female' : 'Male'; 
  const race_names = ["White", "Black", "Asian", "Indian", "Others"];
  const racePred = race_names[raceProb.argMax(-1).dataSync()[0]];

  // 获取画布上下文
  const ctx = canvasRef.getContext('2d');

  // 渲染图片
  ctx.drawImage(source, 0, 0, modelWidth, modelHeight);

   // 设置文本样式
  ctx.font = '12px Arial';
  ctx.fillStyle = 'red';

  // 准备文本内容
  const ageText = `Age: ${Math.round(agePred)}`;
  const genderText = `Gender: ${genderPred}`;
  const raceText = `Race: ${racePred}`;

  // 获取文本的最大宽度
  const maxWidth = Math.max(ctx.measureText(ageText).width, ctx.measureText(genderText).width, ctx.measureText(raceText).width);

  // 在画布上绘制白色半透明背景
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'; 
  ctx.fillRect(5, 5, maxWidth + 10, 60); // 添加 10px 的padding

  // 在画布上绘制文本
  ctx.fillStyle = 'red';
  ctx.fillText(ageText, 10, 20);
  ctx.fillText(genderText, 10, 40);
  ctx.fillText(raceText, 10, 60);

  callback();
};





/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.LayersModel} agemodel loaded Age Prediction tensorflow.js model
 * @param {tf.LayersModel} genmodel loaded Gender Prediction tensorflow.js model
 * @param {tf.LayersModel} racemodel loaded Race Prediction tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export const detectVideo = (vidSource, agemodel, genmodel, racemodel, canvasRef) => {
    /**
     * Function to detect every frame from video
     */
    const detectFrame = async () => {
        if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
            const ctx = canvasRef.getContext("2d");
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
            return; // handle if source is closed
        }
        
        // Detect the entities within the video frame
        await detect(vidSource, agemodel, genmodel, racemodel, canvasRef, () => {
            requestAnimationFrame(detectFrame); // process the next frame
        });
    };

    detectFrame(); // initialize to detect every frame
};



