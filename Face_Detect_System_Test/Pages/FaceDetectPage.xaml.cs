using Emgu.CV.Face;
using Emgu.CV;
using System;
using System.Collections.Generic;
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
using System.Drawing;
using System.IO;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Text.RegularExpressions;
using System.Windows.Threading;
using System.Threading;
using System.Runtime.InteropServices;

namespace Face_Detect_System_Test.Pages
{
    /// <summary>
    /// Логика взаимодействия для FaceDetectPage.xaml
    /// </summary>
    public partial class FaceDetectPage : Page
    {

        // пути к моделям
        private const string pathRecModel = "H:\\mymod.xml"; // путь до модели распознавания лиц (mymod)
        private const string pathYuNetModel = "H:\\face_detection_yunet_2023mar.onnx"; //путь до модели нейронной сети YuNet

        public bool checkWeb = true;
        public bool checkVideo = true;


        private bool CheckHW = true;
        private int frameWidth;
        private int frameHeight;
        private double fps;


        // Создаем распознаватель лиц
        private LBPHFaceRecognizer recognizer = new LBPHFaceRecognizer();
        private FaceDetectorYN _detector;
        private FacesDetect facesDetect = new FacesDetect();

        // Импортируем функцию для удаления HBitmap
        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);

        public FaceDetectPage()
        {
            InitializeComponent();
            VideoFile.Visibility = Visibility.Hidden;
            WebCamStart.IsEnabled = true;
            VideoFileStart.IsEnabled = true;
            recognizer.Read(pathRecModel);
            
        }



        public async void DetectFaceWebCam()
        {
            using (VideoCapture capture = new VideoCapture(0))
            {
                if (recognizer == null)
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
                    PersInfoView.Items.Clear();
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

                    frame = facesDetect.FaceRecognition(frame, faces, recognizer, ref perInf);

                    foreach (PersonInfo cp in perInf)
                    {
                        PersInfoView.Items.Add(cp);
                    }
                    FIOutputImage.Source = BitmapSourceConvert(frame);

                    
                    await Task.Delay(1);

                }

            }
        }

        public async void DetectFaceVideoFile(string vFilePath)
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
                    PersInfoView.Items.Clear();
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

                    frame = facesDetect.FaceRecognition(frame, faces, recognizer, ref perInf);

                    foreach (PersonInfo cp in perInf)
                    {
                        PersInfoView.Items.Add(cp);
                    }
                    FIOutputImage.Source = BitmapSourceConvert(frame);


                    await Task.Delay(1);

                }
                FIOutputImage.Source = new BitmapImage(new Uri("/Images/Default_picture.png", UriKind.Relative));
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

        private void WebCamStart_Click(object sender, RoutedEventArgs e)
        {
            VideoFile.Visibility = Visibility.Hidden;
            WebCamStart.IsEnabled = false;
            VideoFileStart.IsEnabled = true;
            checkVideo = false;
            checkWeb = true;
            CheckHW = true;
            _detector?.Dispose();

            DetectFaceWebCam();
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
            CheckHW = true;
            _detector?.Dispose();
            checkVideo = false;
            var myopenFileDialog = new Microsoft.Win32.OpenFileDialog();
            if (myopenFileDialog.ShowDialog() == true)
            {
                if (myopenFileDialog.FileName.EndsWith(".mp4"))
                {
                    checkVideo = true;
                    DetectFaceVideoFile(myopenFileDialog.FileName);
                }
                else
                {
                    System.Windows.MessageBox.Show("Не верный формат файла!");
                }
            }
        }
    }
}
