//import labels from "./labels.json";

/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvasRef canvas tag reference
 * @param {Array} boxes_data boxes array
 * @param {Array} scores_data scores array
 * @param {Array} classes_data class array
 * @param {Array[Number]} ratios boxes ratio [xRatio, yRatio]
 */
export const renderBoxes = (canvasRef, boxes_data, scores_data, classes_data, ratios) => {
  const ctx = canvasRef.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  const colors = new Colors();
  
// 计算线的位置

  
  // 定义一个对象来存储经过线的车辆总数
  let lineCrossCounts = {
    inCar: 0, // 存储进入的车辆总数
    outCar: 0 // 存储离开的车辆总数
  };

  // font configs
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    12
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";
   
  const classColors = {
  car_plate: "#C71585"
};


  for (let i = 0; i < scores_data.length; ++i) {
    // filter based on class threshold
    const klass = labels[classes_data[i]];
	if (!classColors.hasOwnProperty(klass)) continue;
    const color = classColors[klass];
    const score = (scores_data[i] * 100).toFixed(1);

    let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4);
    x1 *= ratios[0];
    x2 *= ratios[0];
    y1 *= ratios[1];
    y2 *= ratios[1];
    const width = x2 - x1;
    const height = y2 - y1;

    // draw box.
    ctx.fillStyle = Colors.hexToRgba(color, 1);
    ctx.fillRect(x1, y1, width, height);

    // draw border box.
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
    ctx.strokeRect(x1, y1, width, height);

     // Draw the label background.
    ctx.fillStyle = color;
    const textWidth = ctx.measureText(score + "%").width;
	//const textWidth = ctx.measureText(klass + " - " + score + "%").width;
    const textHeight = parseInt(font, 8); // base 10
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText, // handle overflow label box textWidth + ctx.lineWidth
      textWidth,
      textHeight + ctx.lineWidth
    );

    // Draw labels
	// 保存当前的字体设置
	const originalFont = ctx.font;

	// 设置新的字体大小
	ctx.font = "12px Arial"; 
    ctx.fillStyle = "#ffffff";
    ctx.fillText(score + "%", x1 - 1, yText < 0 ? 0 : yText); 
	//ctx.fillText(klass + " - " + score + "%", x1 - 1, yText < 0 ? 0 : yText); 
	ctx.font = originalFont;
	

	
/*  	const centerY = (y1 + y2) / 2;
	const centerX = (x1 + x2) / 2;
    if(centerY >= (crossLineHeight ) && centerX <= crossLineWidth) {
      lineCrossCounts.inCar += 1;
    }
	if(centerY < (crossLineHeight ) && centerX > crossLineWidth) {
      lineCrossCounts.outCar += 1;
    } */
  }
	

// ... (原来的代码)
	// 绘制直线
 	 // 绘制直线
	


	// 定义所有类别和对应的颜色
	const allClasses = [
	  { name: "car_plate", color: "#C71585" }
	];

	// 计算每个类别的数量
	let classCounts = {};
	for (const klass of allClasses) {
	  classCounts[klass.name] = 0; // 初始化每个类别的数量为0
	}
	for (let i = 0; i < classes_data.length; ++i) {
	  const klass = labels[classes_data[i]];
	  if (classCounts.hasOwnProperty(klass)) {
		classCounts[klass]++;
	  }
	}

	// 定义图例的位置和大小
	const legendX = 10; // 图例的 x 坐标
	const legendY = 10; // 图例的 y 坐标
	const boxSize = 20; // 每个颜色框的大小
	//const lineHeight = 28; // 每一行的高度

	// 设置字体和颜色
	ctx.font = 'bold 16px Arial';
	ctx.textBaseline = 'middle';
	
		// 定义图例的位置和大小
	const paddingY = 10; // 文字与上下边框的距离
	const lineHeight = 16 + 1.5 * paddingY; // 每一行的高度为字体大小加上上下边距

	// 计算整个图例的宽度和高度
	let maxWidth = 0;
	const totalHeight = lineHeight * allClasses.length;
	for (const klass of allClasses) {
	  const text = `${klass.name}  ${classCounts[klass.name]}`;
	  const textWidth = ctx.measureText(text).width;
	  maxWidth = Math.max(maxWidth, textWidth);
	}
	const totalWidth = maxWidth + boxSize + 15;

	// 绘制整个图例的背景框
	ctx.fillStyle = 'rgba(255,255,255,0.8)'; // 白色，半透明
	ctx.fillRect(legendX, legendY, totalWidth, totalHeight);

	// 绘制每个类别
	let offsetY = 0;
	for (const klass of allClasses) {
	  const color = klass.color; // 获取类别对应的颜色
	// 绘制颜色框
	  ctx.fillStyle = color;
	  const boxY = legendY + offsetY + (lineHeight - boxSize) / 2; // 计算颜色框的垂直位置
	  ctx.fillRect(legendX, boxY, boxSize, boxSize);

	  // 绘制文字
	  ctx.fillStyle = '#000000'; // 使用深色以提高文字的清晰度
	  const textY = legendY + offsetY + lineHeight / 2; // 计算文字的垂直位置，使其与颜色框的中心对齐
	  ctx.fillText(`${klass.name} : ${classCounts[klass.name]}`, legendX + boxSize + 5, textY); // 文字垂直居中


	  // 更新偏移量，准备绘制下一行
	  offsetY += lineHeight;
	}



};

class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#C71585"
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(
          ", "
        )}, ${alpha})`
      : null;
  };
}
