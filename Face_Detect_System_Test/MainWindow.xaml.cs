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
using System.IO;
using Face_Detect_System_Test.Pages;
using System.Windows.Controls;
using System.Windows.Input;

namespace Face_Detect_System_Test
{
    public partial class MainWindow : Window
    {

        // Определите переменную для хранения текущей страницы
        private Page currentPage;

        private FaceIdentifyPage FIPage;
        private FaceDetectPage FDPage;

        //private FaceDetectorYN _detector;
        private FacesDetect facesDetect = new FacesDetect();
        private ModelTraining modelTr = new ModelTraining();

        // Создаем распознаватель лиц
        private LBPHFaceRecognizer recognizer = new LBPHFaceRecognizer();

        public MainWindow()
        {
            InitializeComponent();
            InitializeFaceDetection();
            
        }

        private void InitializeFaceDetection()
        {
            try
            {
                recognizer.Read("H:\\mymod.xml");
                //_detector = facesDetect.DetectorInit("H:\\face_detection_yunet_2023mar.onnx");

                

            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка при инициализации: {ex.Message}");
                Close();
            }
        }

        private void FIPan_MouseDown(object sender, MouseButtonEventArgs e)
        {
            FIPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 110, 110, 110));
            FIText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 255, 255, 255));
            FIImg.Source = new BitmapImage(new Uri(@"/Images/Face_detect_light.png", UriKind.Relative));

            FDPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(0, 0, 0, 0));
            FDText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 0, 0, 0));
            FDImg.Source = new BitmapImage(new Uri(@"/Images/Face_identify_dark.png", UriKind.Relative));

            // Проверьте, если текущая страница не является
            if (FIPage != null)
                return; // Если уже открыта подходящая страница, то ничего не делаем
            if (FDPage != null)
            {
                FDPage.checkWeb = false;
                FDPage.checkVideo = false;
                FDPage = null;
            }
            currentPage = null;
            FIPage = new FaceIdentifyPage();
            MainFrame.Navigate(FIPage);
            Manager.MainFrame = MainFrame;
        }

        private void FDPan_MouseDown(object sender, MouseButtonEventArgs e)
        {
            FIPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(0, 0, 0, 0));
            FIText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 0, 0, 0));
            FIImg.Source = new BitmapImage(new Uri(@"/Images/Face_detect_dark.png", UriKind.Relative));

            FDPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 110, 110, 110));
            FDText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 255, 255, 255));
            FDImg.Source = new BitmapImage(new Uri(@"/Images/Face_identify_light.png", UriKind.Relative));

            // Проверьте, если текущая страница не является
            if (FDPage != null)
                return; // Если уже открыта подходящая страница, то ничего не делаем
            if (FIPage != null)
            {
                FIPage.checkWeb = false;
                FIPage.checkVideo = false;
                FIPage = null;
            }
            FDPage = new FaceDetectPage();
            MainFrame.Navigate(FDPage);
            Manager.MainFrame = MainFrame;

        }

        


        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);
            //_detector?.Dispose();
            facesDetect?.Dispose();

        }

        private void ModelTren_Click(object sender, RoutedEventArgs e)
        {
            // Подготовка данных для обучения
            string[] trainingImagesPaths = Directory.GetFiles("training_folder", "*.jpg");
            modelTr.ModelTrain("H:\\mymod.xml", trainingImagesPaths, 0);
            Console.WriteLine("Модель обучена!");
        }
    }
}