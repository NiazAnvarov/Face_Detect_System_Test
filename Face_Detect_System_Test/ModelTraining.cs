using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu.CV.Structure;
using System.Security.Cryptography;
using System.Windows.Controls;
using Emgu.CV.Dnn;

namespace Face_Detect_System_Test
{
    internal class ModelTraining
    {

        private LBPHFaceRecognizer recognizer = new LBPHFaceRecognizer();
        private FacesDetect faceDetector = new FacesDetect();
        private FaceDetectorYN _detector;

        public void ModelTrain(string modelPath, string[] trainingImagesPaths, int label)
        {

            

            // Загрузка изображений для обучения
            Mat[] trainingImages = trainingImagesPaths.Select(path => CvInvoke.Imread(path, ImreadModes.Grayscale)).ToArray();
            Mat face = new Mat();
            List<Mat> images = new List<Mat>();
            List<int> labels = new List<int>(); ;

            foreach (Mat frame in trainingImages)
            {

                _detector = new FaceDetectorYN(
                    model: "H:\\face_detection_yunet_2023mar.onnx",
                    config: string.Empty,
                    inputSize: new System.Drawing.Size(frame.Cols, frame.Rows),
                    scoreThreshold: 0.9f,
                    nmsThreshold: 0.3f,
                    topK: 5000,
                    backendId: Emgu.CV.Dnn.Backend.Default,
                    targetId: Target.Cpu);

                CvInvoke.CvtColor(frame, frame, ColorConversion.Gray2Bgr);

                face = faceDetector.DetectFaces(frame, _detector);

                if (face.Rows > 0)
                {
                    var faceData = new Matrix<float>(face.Rows, face.Cols);
                    face.CopyTo(faceData);

                    for (int i = 0; i < face.Rows; i++)
                    {
                        float confidence = faceData[i, 0];
                        if (confidence >= 0.9f)
                        {
                            // Нормализация координат центра
                            float centerX = faceData[i, 4] + faceData[i, 2] / 4;
                            float centerY = faceData[i, 1] + faceData[i, 3] / 4;

                            // Нормализация размеров
                            float width = faceData[i, 2] * (float)1.1;
                            float height = faceData[i, 3] * (float)1.1;

                            int frameWidth = frame.Width;
                            int frameHeight = frame.Height;

                            // Преобразование в пиксели с учетом размера кадра
                            int rectX = (int)(centerX * frameWidth - width * frameWidth / 2);
                            int rectY = (int)(centerY * frameHeight - height * frameHeight / 2);
                            int rectWidth = (int)(width);
                            int rectHeight = (int)(height);

                            // Ограничение по границам изображения
                            rectX = (int)(centerX - width / 1.9);
                            rectY = (int)(centerY - height / 3.8); 

                            //Если рамка выходит за границы кадра
                            if (rectY + rectHeight > frame.Height)
                            {
                                rectHeight -= rectY + rectHeight - frame.Height;
                            }
                            if (rectY < 0)
                            {
                                rectHeight += rectY;
                                rectY = 0;
                            }

                            if (rectX + rectWidth > frame.Width)
                            {
                                rectWidth -= rectX + rectWidth - frame.Width;
                            }
                            if (rectX < 0)
                            {
                                rectWidth += rectX;
                                rectX = 0;
                            }

                            // Обрезаем область лица из кадра
                            Rectangle faceRect = new Rectangle(rectX, rectY, rectWidth, rectHeight);
                            // Обрезаем лицо
                            Mat faceImage = new Mat(frame, faceRect);

                            // Конвертируем в черно-белое изображение
                            Mat grayFace = new Mat();

                            CvInvoke.CvtColor(faceImage, grayFace, ColorConversion.Bgr2Gray);
                            CvInvoke.EqualizeHist(grayFace, grayFace);

                            images.Add(grayFace);
                            labels.Add(label);
                        }
                    }
                }

                _detector?.Dispose();
            }

            

            if (images.Count > 0)
            {
                
                recognizer.Train(images.ToArray(), labels.ToArray());
                recognizer.Write(modelPath);
            }
        }

    }
}
