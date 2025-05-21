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
using System.Windows.Media.Imaging;
using System.Windows;

namespace Face_Detect_System_Test
{
    internal class ModelTraining
    {

        private readonly Manager _settingsManager = new Manager();
        private FacesDetect faceDetector = new FacesDetect();
        private FaceDetectorYN _detector;

        public void ModelTrain(string modelPath, List<Mat> Faces, int label)
        {


            // Загрузка изображений для обучения
            List<Mat> images = new List<Mat>();
            List<int> labels = new List<int>();
            Mat grayFace = new Mat();

            foreach (Mat face in Faces)
            {
                CvInvoke.CvtColor(face, grayFace, ColorConversion.Bgr2Gray);
                CvInvoke.EqualizeHist(grayFace, grayFace);

                images.Add(grayFace);
                labels.Add(label);
            }

            if (images.Count > 0)
            {
                if (Manager.RecognizerModelPath != null)
                {
                    //recognizer.Read(modelPath);
                    //recognizer.Train(images.ToArray(), labels.ToArray());
                    Manager.recognizer.Update(images.ToArray(), labels.ToArray());
                    Manager.recognizer.Write(modelPath);
                }
                else
                {

                    modelPath = "H:\\newRecMod.xml";
                    Manager.recognizer.Train(images.ToArray(), labels.ToArray());
                    Manager.recognizer.Write(modelPath);
                    Manager.RecognizerModelPath = modelPath;
                    _settingsManager.SaveModelPath(modelPath);
                }
            }
        }

        public List<Mat> FacesDetect(string vFilePath, String pathYuNetModel)
        {
            bool checkWH = true;
            List<Mat> facesFromFrame = new List<Mat>();
            Mat frame = new Mat();
            Mat faces = new Mat();

            using (VideoCapture capture = new VideoCapture(vFilePath))
            {
                if (!capture.IsOpened)
                {
                    _detector?.Dispose();
                    return null;
                }

                // Получаем параметры видео

                while (true)
                {
                    if (!capture.Read(frame))
                        break;
                    
                    if (frame.IsEmpty)
                        break;

                    if (checkWH)
                    {
                        int frameWidth = frame.Width;
                        int frameHeight = frame.Height;
                        _detector = faceDetector.DetectorInit(pathYuNetModel, frameWidth, frameHeight);
                        checkWH = false;
                    }
                    CvInvoke.Flip(frame, frame, FlipType.Horizontal);
                    faces = faceDetector.DetectFaces(frame, _detector);

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

                                    int rectX = (int)(centerX * frame.Width - width * frame.Width / 2);
                                    int rectY = (int)(centerY * frame.Height - height * frame.Height / 2);
                                    int rectWidth = (int)(width);
                                    int rectHeight = (int)(height);

                                    // Ограничение по границам изображения
                                    rectX = (int)(centerX - width / 1.9);
                                    rectY = (int)(centerY - height / 3.8);

                                    // Если рамка выходит за границы кадра
                                    if (rectY + rectHeight > frame.Height)
                                        rectHeight -= rectY + rectHeight - frame.Height;
                                    if (rectY < 0)
                                    {
                                        rectHeight += rectY;
                                        rectY = 0;
                                    }

                                    if (rectX + rectWidth > frame.Width)
                                        rectWidth -= rectX + rectWidth - frame.Width;
                                    if (rectX < 0)
                                    {
                                        rectWidth += rectX;
                                        rectX = 0;
                                    }

                                    Rectangle faceRect = new Rectangle(rectX, rectY, rectWidth, rectHeight);
                                    Mat faceImage = new Mat(frame, faceRect);
                                    facesFromFrame.Add(faceImage);
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                    }
                }
            }

            return facesFromFrame;
        }

        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);

        private BitmapSource BitmapSourceConvert(Mat mat)
        {
            if (mat.IsEmpty)
                throw new ArgumentException("Source Mat is empty.");

            using (var bitmap = mat.ToImage<Bgr, byte>().ToBitmap())
            {
                var hBitmap = bitmap.GetHbitmap();
                try
                {
                    return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                        hBitmap,
                        IntPtr.Zero,
                        Int32Rect.Empty,
                        BitmapSizeOptions.FromEmptyOptions());
                }
                finally
                {
                    DeleteObject(hBitmap);
                }
            }
        }

    }
}
