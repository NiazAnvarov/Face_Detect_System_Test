using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Features2D;
using Emgu.CV.Ocl;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Data.Entity.Core.Metadata.Edm;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using static System.Net.Mime.MediaTypeNames;

namespace Face_Detect_System_Test
{
    internal class FacesDetect
    {

        private FaceDetectorYN _detector;
        private int Distance = 55;
        private AesEncryption aesEncryption = new AesEncryption(Manager.key, Manager.iv);

        public FaceDetectorYN DetectorInit(string modelPath, int width, int height)
        {
            _detector = new FaceDetectorYN(
                    model: modelPath,
                    config: string.Empty,
                    inputSize: new System.Drawing.Size(width, height),
                    scoreThreshold: 0.9f,
                    nmsThreshold: 0.3f,
                    topK: 5000,
                    backendId: Emgu.CV.Dnn.Backend.Default,
                    targetId: Target.Cpu);
            return _detector;
        }

        public void Dispose()
        {
            _detector?.Dispose();
        }

        public Mat DetectFaces(Mat frame, FaceDetectorYN _detector)
        {
            var faces = new Mat();
            
            _detector.Detect(frame, faces);

            return faces;
        }

        public Mat FaceRecognition(Mat frame, Mat faces, LBPHFaceRecognizer recognizer, ref List<PersonInfo> perInfo)
        {
            try
            {
                if (faces.Rows > 0)
                {
                    
                    var facesData = new Matrix<float>(faces.Rows, faces.Cols);
                    faces.CopyTo(facesData);


                    for (int i = 0; i < faces.Rows; i++)
                    {
                        float confidence = facesData[i, 0];
                        if (confidence >= 0.9f)
                        {
                            // Нормализация координат центра
                            float centerX = facesData[i, 4] + facesData[i, 2] / 4;
                            float centerY = facesData[i, 1] + facesData[i, 3] / 4;

                            // Нормализация размеров
                            float width = facesData[i, 2] * (float)1.1;
                            float height = facesData[i, 3] * (float)1.1;

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
                            PointF eyeLeft = new PointF((int)facesData[i, 4], (int)facesData[i, 5]);
                            PointF eyeRight = new PointF((int)facesData[i, 6], (int)facesData[i, 7]);

                            // Рассчитаем центр между глазами
                            PointF eyecenter = new PointF((eyeLeft.X + eyeRight.X) / 2, (eyeLeft.Y + eyeRight.Y) / 2);

                            // Вычислим угол наклона
                            var deltaY = eyeRight.Y - eyeLeft.Y;
                            var deltaX = eyeRight.X - eyeLeft.X;
                            var angle = Math.Atan2(deltaY, deltaX) * (180.0 / Math.PI);

                            // Определим желаемую высоту и ширину
                            int desiredWidth = frame.Width;  // Можно выбрать нужную ширину
                            int desiredHeight = frame.Height; // Можно выбрать нужную высоту

                            // Создайте матрицу преобразования
                            Mat matrix = new Mat();
                            CvInvoke.GetRotationMatrix2D(eyecenter, (float)angle, 1, matrix);

                            // Примените аффинное преобразование
                            Mat alignedFace = new Mat();
                            CvInvoke.WarpAffine(frame, alignedFace, matrix, new System.Drawing.Size(desiredWidth, desiredHeight), Inter.Linear, Warp.Default);

                            // Обрезаем лицо
                            Mat faceImage = new Mat(alignedFace, faceRect);
                            
                            // Конвертируем в черно-белое изображение и нормализуем размер
                            Mat grayFace = new Mat();
                            //CvInvoke.CvtColor(faceImage, grayFace, ColorConversion.Bgr2Gray);
                            grayFace = ProcessImage(faceImage);
                            CvInvoke.EqualizeHist(grayFace, grayFace);

                            // Распознаем человека на изображении
                            var result = recognizer.Predict(grayFace);
                            Console.WriteLine(result.Distance);
                            if (result.Distance < Distance)
                            {
                                string displayText = "";
                                int predictedLabel = result.Label;
                                var currentPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.Where(p => p.ID == predictedLabel).ToList();
                                foreach (var cp in currentPerson)
                                {
                                    displayText = "";
                                    displayText = aesEncryption.Decrypt(cp.LastName).ToString() + " " + aesEncryption.Decrypt(cp.FirstName).ToString() + " " + aesEncryption.Decrypt(cp.Patronymic).ToString();
                                    perInfo.Add(cp);
                                }

                                // Рисуем прямоугольник и результат распознавания
                                CvInvoke.Rectangle(frame, faceRect, new MCvScalar(0, 255, 0), 2); // Зеленый для распознанных лиц 
                                                                                                  // Отображаем результат распознавания
                                CvInvoke.PutText(frame, displayText,
                                    new System.Drawing.Point(rectX, rectY - 10),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex,
                                    1,
                                    new MCvScalar(0, 255, 0));
                            }
                            else
                            {
                                // Рисуем прямоугольник и результат распознавания
                                CvInvoke.Rectangle(frame, faceRect, new MCvScalar(0, 0, 255), 2); // Красный для распознанных лиц 
                                                                                                  // Отображаем результат распознавания
                                CvInvoke.PutText(frame,
                                    "Unknown",
                                    new System.Drawing.Point(rectX, rectY - 10),
                                    Emgu.CV.CvEnum.FontFace.HersheySimplex,
                                    1,
                                    new MCvScalar(0, 0, 255));
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            
            return frame;
        }

        public Mat FaceIdentify(Mat frame, Mat faces, LBPHFaceRecognizer recognizer, ref List<PersonInfo> perInfo, String FIOSearch)
        {

            try
            {

                if (faces.Rows > 0)
                {
                    var facesData = new Matrix<float>(faces.Rows, faces.Cols);
                    faces.CopyTo(facesData);


                    for (int i = 0; i < faces.Rows; i++)
                    {
                        float confidence = facesData[i, 0];
                        if (confidence >= 0.9f)
                        {
                            // Нормализация координат центра
                            float centerX = facesData[i, 4] + facesData[i, 2] / 4;
                            float centerY = facesData[i, 1] + facesData[i, 3] / 4;

                            // Нормализация размеров
                            float width = facesData[i, 2] * (float)1.1;
                            float height = facesData[i, 3] * (float)1.1;

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
                            PointF eyeLeft = new PointF((int)facesData[i, 4], (int)facesData[i, 5]);
                            PointF eyeRight = new PointF((int)facesData[i, 6], (int)facesData[i, 7]);

                            // Рассчитаем центр между глазами
                            PointF eyecenter = new PointF((eyeLeft.X + eyeRight.X) / 2, (eyeLeft.Y + eyeRight.Y) / 2);

                            // Вычислим угол наклона
                            var deltaY = eyeRight.Y - eyeLeft.Y;
                            var deltaX = eyeRight.X - eyeLeft.X;
                            var angle = Math.Atan2(deltaY, deltaX) * (180.0 / Math.PI);

                            // Определим желаемую высоту и ширину
                            int desiredWidth = frame.Width;  // Можно выбрать нужную ширину
                            int desiredHeight = frame.Height; // Можно выбрать нужную высоту

                            // Создайте матрицу преобразования
                            Mat matrix = new Mat();
                            CvInvoke.GetRotationMatrix2D(eyecenter, (float)angle, 1, matrix);

                            // Примените аффинное преобразование
                            Mat alignedFace = new Mat();
                            CvInvoke.WarpAffine(frame, alignedFace, matrix, new System.Drawing.Size(desiredWidth, desiredHeight), Inter.Linear, Warp.Default);

                            // Обрезаем лицо
                            Mat faceImage = new Mat(alignedFace, faceRect);

                            // Конвертируем в черно-белое изображение и нормализуем размер
                            Mat grayFace = new Mat();
                            //CvInvoke.CvtColor(faceImage, grayFace, ColorConversion.Bgr2Gray);
                            grayFace = ProcessImage(faceImage);
                            CvInvoke.EqualizeHist(grayFace, grayFace);

                            // Распознаем человека на изображении
                            var result = recognizer.Predict(grayFace);
                            Console.WriteLine(result.Distance);
                            
                            if (result.Distance < Distance)
                            {
                                //FIOSearch = aesEncryption.Encrypt(FIOSearch);
                                string displayText = "";
                                int predictedLabel = result.Label;
                                var currentPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.Where(p => p.ID == predictedLabel).ToList();
                                currentPerson = currentPerson.Where(p => FIOSearch.ToLower().Contains(aesEncryption.Decrypt(p.LastName.ToString()).ToLower()) || FIOSearch.ToLower().Contains(aesEncryption.Decrypt(p.FirstName.ToString()).ToLower()) || FIOSearch.ToLower().Contains(aesEncryption.Decrypt(p.Patronymic.ToString()).ToLower())).ToList();
                                if (currentPerson.Count > 0)
                                {
                                    foreach (var cp in currentPerson)
                                    {
                                        displayText = "";
                                        displayText = aesEncryption.Decrypt(cp.LastName.ToString()) + " " + aesEncryption.Decrypt(cp.FirstName.ToString()) + " " + aesEncryption.Decrypt(cp.Patronymic.ToString());
                                        perInfo.Add(cp);
                                    }

                                    // Рисуем прямоугольник и результат распознавания
                                    CvInvoke.Rectangle(frame, faceRect, new MCvScalar(0, 255, 0), 2); // Зеленый для распознанных лиц 
                                                                                                      // Отображаем результат распознавания
                                    CvInvoke.PutText(frame, displayText,
                                        new System.Drawing.Point(rectX, rectY - 10),
                                        Emgu.CV.CvEnum.FontFace.HersheySimplex,
                                        1,
                                        new MCvScalar(0, 255, 0));
                                }
                            }


                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            return frame;
        }

        public Mat ProcessImage(Mat inputImage)
        {
            // Шаг 1: Предобработка изображения
            Mat preprocessedImage = PreprocessImage(inputImage);

            // Шаг 3: Устранение теней/бликов
            preprocessedImage = RemoveShadows(preprocessedImage);

            return preprocessedImage;
        }

        public Mat PreprocessImage(Mat inputImage)
        {

            // Шаг 1: Нормализация освещения с использованием CLAHE
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(inputImage, grayImage, ColorConversion.Bgr2Gray);
            Mat claheImage = new Mat();
            // Создаем объект CLAHE
            CvInvoke.CLAHE(grayImage, 2.0, new Size(8, 8), claheImage);

            // Шаг 2: Увеличение контрастности (можно дополнительно настроить)
            Mat contrastImage = new Mat();
            CvInvoke.Normalize(claheImage, contrastImage, 0, 255, NormType.MinMax, DepthType.Cv8U);

            return contrastImage;
        }

        public Mat RemoveShadows(Mat inputImage)
        {
            // Примените фильтр Гаусса для сглаживания изображения
            Mat smoothedImage = new Mat();
            CvInvoke.GaussianBlur(inputImage, smoothedImage, new Size(5, 5), 0);

            return smoothedImage;
        }


    }
}
