using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Emgu.CV.Face;
using Emgu.CV.Features2D;
using Emgu.CV.Util;

namespace Face_Detect_System_Test.Pages
{
    /// <summary>
    /// Логика взаимодействия для FaceIdentifyPage.xaml
    /// </summary>
    public partial class FaceIdentifyPage : Page
    {

        // пути к моделям
        
        
        private const string pathYuNetModel = "H:\\face_detection_yunet_2023mar.onnx"; //путь до модели нейронной сети YuNet

        public bool checkWeb = true;
        public bool checkVideo = true;

        private Mat facePhoto = new Mat();


        private bool CheckHW = true;
        private int frameWidth;
        private int frameHeight;
        private double fps;

        // Создаем распознаватель лиц
        
        
        private FaceDetectorYN _detector;
        private FacesDetect facesDetect = new FacesDetect();
        

        public FaceIdentifyPage()
        {
            InitializeComponent();
            VideoFile.Visibility = Visibility.Hidden;
            WebCamStart.IsEnabled = true;
            VideoFileStart.IsEnabled = true;            
        }

        public static ImageSource BitmapToImageSource(Bitmap bitmap)
        {
            using (MemoryStream memoryStream = new MemoryStream())
            {
                // Сохранение Bitmap в MemoryStream в формате PNG
                bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
                memoryStream.Position = 0; // Сброс позиции потока на начало

                // Создание BitmapImage из MemoryStream
                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memoryStream;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad; // Сохраняем данные в памяти
                bitmapImage.EndInit();
                bitmapImage.Freeze(); // Замораживание для использования из другого потока, если необходимо

                return bitmapImage; // Возвращаем ImageSource
            }
        }

        public async void IdentifyFaceWebCam()
        {
            using (VideoCapture capture = new VideoCapture(0))
            {

                if (Manager.recognizer == null)
                {
                    // Модель не была загружена правильно
                    throw new Exception("Модель не была загружена или обучена.");
                }

                // Получение необходимых параметров видео
                if (CheckHW)
                {
                    frameWidth = (int)capture.Get(CapProp.FrameWidth);
                    frameHeight = (int)capture.Get(CapProp.FrameHeight);
                    fps = capture.Get(CapProp.Fps);
                    _detector = facesDetect.DetectorInit(pathYuNetModel, frameWidth, frameHeight);
                    CheckHW = false;
                }

                Mat frame = new Mat(); // Для хранения каждого кадра


                while (checkWeb)
                {

                    // Чтение следующего кадра видео
                    capture.Read(frame);
                    if (frame.IsEmpty)
                    {
                        _detector?.Dispose();
                        CheckHW = true;
                        FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
                        Console.WriteLine("Процесс окончен!");
                        break; // Выход, если кадры закончились
                    }
                    CvInvoke.Flip(frame, frame, FlipType.Horizontal); // Отзеркаливание по горизонтали

                    var faces = new Mat();
                    faces = facesDetect.DetectFaces(frame, _detector);



                    var perInf = new List<PersonInfo>();
                    frame = facesDetect.FaceIdentify(frame, faces, Manager.recognizer, ref perInf, FIOSearch.Text);


                    FIOutputImage.Source = BitmapSourceConvert(frame);
                    await Task.Delay(1);

                }
            }
        }

        public async void IdentifyFaceVideoFile( string vFilePath)
        {
            using (VideoCapture capture = new VideoCapture(vFilePath))
            {

                if (!capture.IsOpened)
                {
                    return;
                }

                // Получение необходимых параметров видео
                if (CheckHW)
                {
                    frameWidth = (int)capture.Get(CapProp.FrameWidth);
                    frameHeight = (int)capture.Get(CapProp.FrameHeight);
                    fps = capture.Get(CapProp.Fps);
                    _detector = facesDetect.DetectorInit(pathYuNetModel, frameWidth, frameHeight);
                    CheckHW = false;
                }

                Mat frame = new Mat(); // Для хранения каждого кадра

                while (checkVideo)
                {

                    // Чтение следующего кадра видео
                    capture.Read(frame);
                    if (frame.IsEmpty)
                    {
                        _detector?.Dispose();
                        CheckHW = true;
                        FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
                        Console.WriteLine("Процесс окончен!");
                        break; // Выход, если кадры закончились
                    }
                    CvInvoke.Flip(frame, frame, FlipType.Horizontal); // Отзеркаливание по горизонтали

                    var faces = new Mat();
                    faces = facesDetect.DetectFaces(frame, _detector);



                    var perInf = new List<PersonInfo>();
                    frame = facesDetect.FaceIdentify(frame, faces, Manager.recognizer, ref perInf, FIOSearch.Text);


                    FIOutputImage.Source = BitmapSourceConvert(frame);
                    await Task.Delay(1);

                }
                FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
            }
        }

        private void WebCamStart_Click(object sender, RoutedEventArgs e)
        {
            _detector?.Dispose();
            CheckHW = true;
            FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
            VideoFile.Visibility = Visibility.Hidden;
            WebCamStart.IsEnabled = false;
            VideoFileStart.IsEnabled = true;
            checkVideo = false;
            checkWeb = true;
            IdentifyFaceWebCam();
        }

        private void VideoFileStart_Click(object sender, RoutedEventArgs e)
        {
            _detector?.Dispose();
            CheckHW = true;
            FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
            checkWeb = false;
            checkVideo = true;
            VideoFileStart.IsEnabled = false;
            WebCamStart.IsEnabled = true;
            VideoFile.Visibility = Visibility.Visible;
            FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
        }

        private void VideoFile_Click(object sender, RoutedEventArgs e)
        {
            _detector?.Dispose();
            CheckHW = true;
            FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
            checkVideo = false;
            var myopenFileDialog = new Microsoft.Win32.OpenFileDialog();
            if (myopenFileDialog.ShowDialog() == true)
            {
                checkVideo = true;
                IdentifyFaceVideoFile(myopenFileDialog.FileName);
            }
        }

        // Метод для преобразования Mat в BitmapSource
        private BitmapSource BitmapSourceConvert(Mat mat)
        {
            // Проверяем, что матрица не пуста
            if (mat.IsEmpty)
                throw new ArgumentException("Source Mat is empty.");

            // Получаем HBitmap
            var bitmap = mat.ToImage<Bgr, byte>().ToBitmap();
            var hBitmap = bitmap.GetHbitmap();

            // Создаем BitmapSource
            var bitmapSource = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                hBitmap,
                IntPtr.Zero,
                Int32Rect.Empty,
                System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());

            // Освобождаем HBitmap
            DeleteObject(hBitmap);

            return bitmapSource;
        }

        // Импортируем функцию для удаления HBitmap
        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);

        

        

    }
}
