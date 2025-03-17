using System;
using System.Windows;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Drawing;
using System.Windows.Threading;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;

namespace Face_Detect_System_Test
{
    public partial class MainWindow : Window
    {
        
        private VideoCapture _capture;
        private DispatcherTimer _timer;
        private readonly object _lock = new object();
        private FacesDetect facesDetect = new FacesDetect();

        // Создаем распознаватель лиц
        private LBPHFaceRecognizer recognizer = new LBPHFaceRecognizer();

        [DllImport("gdi32")]
        private static extern int DeleteObject(IntPtr o);

        public MainWindow()
        {
            InitializeComponent();
            InitializeFaceDetection();
            recognizer.Read("H:\\rec.xml");
        }

        private void InitializeFaceDetection()
        {
            try
            {
                _capture = new VideoCapture(0);
                _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
                _timer.Tick += Timer_Tick;
                _timer.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка при инициализации: {ex.Message}");
                Close();
            }
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            lock (_lock)
            {
                try
                {
                    var frame = _capture.QueryFrame();
                    if (frame == null) return;

                    var faces = new Mat();
                    faces = facesDetect.DetectFaces(frame);
                    
                    try
                    {
                        if (faces.Rows > 0)
                        {
                            var facesData = new Matrix<float>(faces.Rows, faces.Cols);
                            faces.CopyTo(facesData);
                            //for(int i = 0; i < faces.Cols; i++)
                            //{
                            //    Console.WriteLine(facesData[0, i]);
                            //}

                            for (int i = 0; i < faces.Rows; i++)
                            {
                                float confidence = facesData[i, 0];
                                if (confidence >= 0.9f)
                                {
                                    // Нормализация координат центра
                                    float centerX = facesData[i, 4] + facesData[i, 2] / 4;
                                    float centerY = facesData[i, 1] + facesData[i, 3] / 4;

                                    // Нормализация размеров
                                    float width = facesData[i, 2];
                                    float height = facesData[i, 3];

                                    int frameWidth = frame.Width;
                                    int frameHeight = frame.Height;

                                    // Преобразование в пиксели с учетом размера кадра
                                    int rectX = (int)(centerX * frameWidth - width * frameWidth / 2);
                                    int rectY = (int)(centerY * frameHeight - height * frameHeight / 2);
                                    int rectWidth = (int)width;
                                    int rectHeight = (int)height;

                                    // Ограничение по границам изображения
                                    rectX = (int)(centerX - width/1.8);
                                    rectY = (int)(centerY - height/3.8);

                                    // Вывод отладочной информации
                                    Console.WriteLine($"Координаты лица:");
                                    Console.WriteLine($"centerX: {centerX:F4}, centerY: {centerY:F4}");
                                    Console.WriteLine($"width: {width:F4}, height: {height:F4}");
                                    Console.WriteLine($"rectX: {rectX}, rectY: {rectY}");
                                    Console.WriteLine($"rectWidth: {rectWidth}, rectHeight: {rectHeight}");

                                    // Обрезаем область лица из кадра
                                    Rectangle faceRect = new Rectangle(rectX, rectY, rectWidth, rectHeight);
                                    // Обрезаем лицо
                                    Mat faceImage = new Mat(frame, faceRect);

                                    // Конвертируем в черно-белое изображение и нормализуем размер
                                    Mat grayFace = new Mat();
                                    Mat resizedFace = new Mat();
                                    CvInvoke.CvtColor(faceImage, grayFace, ColorConversion.Bgr2Gray);
                                    CvInvoke.Resize(grayFace, resizedFace, new System.Drawing.Size(250, 250));


                                    // Распознаем человека на изображении
                                    var result = recognizer.Predict(grayFace);
                                    int predictedLabel = result.Label;
                                    float confidenceRez = (float)result.Distance;

                                    // Рисуем прямоугольник и результат распознавания
                                    CvInvoke.Rectangle(frame,
                                        faceRect,
                                        result.Label < 60 ?
                                            new MCvScalar(0, 0, 255) : // Красный для неизвестных лиц
                                            new MCvScalar(0, 255, 0), // Зеленый для распознанных лиц
                                        2);
                                    //CvInvoke.Flip(frame, frame, FlipType.Horizontal); // Отзеркаливание по горизонтали
                                    //// Отображаем результат распознавания
                                    //CvInvoke.PutText(frame,
                                    //    $"ID: {result.Label}, Confidence: {result.Distance:F2}",
                                    //    new System.Drawing.Point(rectY, rectX - 10),
                                    //    Emgu.CV.CvEnum.FontFace.HersheySimplex,
                                    //    1, 
                                    //    new MCvScalar(0, 255, 0));
                                }
                            }
                        }
                    }
                    catch(Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                    }

                    ImgOut.Source = BitmapSourceConvert(frame);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка при обработке кадра: {ex.Message}");
                }
            }
        }

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



        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);
            _timer?.Stop();
            _capture?.Dispose();
            facesDetect?.Dispose();

        }
    }
}