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

        // Создаем распознаватель лиц

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
                            // Обрезаем лицо
                            Mat faceImage = new Mat(frame, faceRect);

                            // Конвертируем в черно-белое изображение и нормализуем размер
                            Mat grayFace = new Mat();

                            CvInvoke.CvtColor(faceImage, grayFace, ColorConversion.Bgr2Gray);
                            CvInvoke.EqualizeHist(grayFace, grayFace);

                            // Распознаем человека на изображении
                            var result = recognizer.Predict(grayFace);
                            Console.WriteLine(result.Distance);

                            if (result.Distance < 70)
                            {
                                string displayText = "";
                                int predictedLabel = result.Label;
                                var currentPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.Where(p => p.ID == predictedLabel).ToList();
                                foreach (var cp in currentPerson)
                                {
                                    displayText = "";
                                    displayText = result.Distance + cp.LastName.ToString() + " " + cp.FirstName.ToString() + " " + cp.Patronymic.ToString();
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
                            // Обрезаем лицо
                            Mat faceImage = new Mat(frame, faceRect);

                            // Конвертируем в черно-белое изображение и нормализуем размер
                            Mat grayFace = new Mat();

                            CvInvoke.CvtColor(faceImage, grayFace, ColorConversion.Bgr2Gray);
                            CvInvoke.EqualizeHist(grayFace, grayFace);

                            // Распознаем человека на изображении
                            var result = recognizer.Predict(grayFace);
                            Console.WriteLine(result.Distance);

                            if (result.Distance < 70)
                            {
                                string displayText = "";
                                int predictedLabel = result.Label;
                                var currentPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.Where(p => p.ID == predictedLabel).ToList();
                                currentPerson = currentPerson.Where(p => FIOSearch.ToLower().Contains(p.LastName.ToLower()) || FIOSearch.ToLower().Contains(p.FirstName.ToLower()) || FIOSearch.ToLower().Contains(p.Patronymic.ToLower())).ToList();
                                if (currentPerson.Count > 0)
                                {

                                    foreach (var cp in currentPerson)
                                    {
                                        displayText = "";
                                        displayText = result.Distance + cp.LastName.ToString() + " " + cp.FirstName.ToString() + " " + cp.Patronymic.ToString();
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

        public Mat FacePhotoIdentify(Mat frame, Mat faces, Mat facePhoto)
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
                            // Обрезаем лицо
                            Mat faceImage = new Mat(frame, faceRect);
                            if (CompareFaces(faceImage, facePhoto))
                            {
                                // Рисуем прямоугольник и результат распознавания
                                CvInvoke.Rectangle(frame, faceRect, new MCvScalar(0, 255, 0), 2); // Зеленый для распознанных лиц 
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

        private bool CompareFaces(Mat img1, Mat img2)
        {


            // Инициализация SIFT
            var sift = new SIFT(500);

            // Извлечение дескрипторов
            VectorOfKeyPoint keypoints1 = new VectorOfKeyPoint();
            Mat descriptors1 = new Mat();
            sift.DetectAndCompute(img1, null, keypoints1, descriptors1, false);

            VectorOfKeyPoint keypoints2 = new VectorOfKeyPoint();
            Mat descriptors2 = new Mat();
            sift.DetectAndCompute(img2, null, keypoints2, descriptors2, false);

            // Сопоставление дескрипторов
            var matcher = new BFMatcher(DistanceType.L2);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            matcher.Add(descriptors2);
            matcher.KnnMatch(descriptors1, matches, 2);

            // Фильтрация совпадений
            int goodMatches = 0;
            for (int i = 0; i < matches.Size; i++)
            {
                var match = matches[i];
                // Приемлемое соотношение - первую пару достаточно хороша, по сравнению со второй
                if (match.Size >= 2 && match[0].Distance < 0.75 * match[1].Distance)
                {
                    goodMatches++;
                }
            }

            // Устанавливаем порог для решения, похожи ли лица
            return goodMatches > 5; // Например, мы считаем, что 5 хороших совпадений достаточно для положительного ответа

        }

    }
}
