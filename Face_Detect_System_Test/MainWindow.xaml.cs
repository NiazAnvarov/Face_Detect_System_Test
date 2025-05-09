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

        private FaceIdentifyPage FIPage;
        private FaceDetectPage FDPage;
        private ModelTrainingPage MTPage;
        private readonly Manager _settingsManager;

        //private FaceDetectorYN _detector;
        private FacesDetect facesDetect = new FacesDetect();
        private ModelTraining modelTr = new ModelTraining();

        public MainWindow()
        {
            InitializeComponent();
            
            _settingsManager = new Manager();

            //Загружаем сохрненный путь при запуске
            string savedPath = _settingsManager.LoadModelPath();
            if (!string.IsNullOrEmpty(savedPath))
            {
                FIPan.IsEnabled = true;
                FDPan.IsEnabled = true;
                FIPan_MouseDown(null, null);
                ModelPathTextBox.Text = savedPath;
                Manager.RecognizerModelPath = savedPath;
                Manager.recognizer.Read(Manager.RecognizerModelPath);
            }
            else
            {
                
                ModelPathTextBox.Text = "Модель не загружена";
                FIPan.IsEnabled = false;
                FDPan.IsEnabled = false;
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

            MTPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(0, 0, 0, 0));
            MTText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 0, 0, 0));
            MTImg.Source = new BitmapImage(new Uri(@"/Images/Model_training_dark.png", UriKind.Relative));

            // Проверьте, если текущая страница не является
            if (FIPage != null)
                return; // Если уже открыта подходящая страница, то ничего не делаем
            if (FDPage != null)
            {
                FDPage.checkWeb = false;
                FDPage.checkVideo = false;
                FDPage = null;
            }
            if(MTPage != null)
            {
                MTPage = null;
            }
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

            MTPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(0, 0, 0, 0));
            MTText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 0, 0, 0));
            MTImg.Source = new BitmapImage(new Uri(@"/Images/Model_training_dark.png", UriKind.Relative));

            // Проверьте, если текущая страница не является
            if (FDPage != null)
                return; // Если уже открыта подходящая страница, то ничего не делаем
            if (FIPage != null)
            {
                FIPage.checkWeb = false;
                FIPage.checkVideo = false;
                FIPage = null;
            }
            if (MTPage != null)
            {
                MTPage = null;
            }
            FDPage = new FaceDetectPage();
            MainFrame.Navigate(FDPage);
            Manager.MainFrame = MainFrame;

        }

        private void MTPan_MouseDown(object sender, MouseButtonEventArgs e)
        {
            FIPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(0, 0, 0, 0));
            FIText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 0, 0, 0));
            FIImg.Source = new BitmapImage(new Uri(@"/Images/Face_detect_dark.png", UriKind.Relative));

            FDPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(0, 0, 0, 0));
            FDText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 0, 0, 0));
            FDImg.Source = new BitmapImage(new Uri(@"/Images/Face_identify_dark.png", UriKind.Relative));

            MTPan.Background = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 110, 110, 110));
            MTText.Foreground = new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 255, 255, 255));
            MTImg.Source = new BitmapImage(new Uri(@"/Images/Model_training_light.png", UriKind.Relative));

            // Проверьте, если текущая страница не является ModelTrainingPage
            if (MTPage != null)
                return; // Если уже открыта подходящая страница, то ничего не делаем
            if (FIPage != null)
            {
                FIPage.checkWeb = false;
                FIPage.checkVideo = false;
                FIPage = null;
            }
            if (FDPage != null)
            {
                FDPage.checkWeb = false;
                FDPage.checkVideo = false;
                FDPage = null;
            }


            // Навигация на ModelTrainingPage
            MTPage = new ModelTrainingPage();
            MainFrame.Navigate(MTPage);
            Manager.MainFrame = MainFrame;
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);
            //_detector?.Dispose();
            facesDetect?.Dispose();

        }

        private void RecModelLoadBtn_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog openFileDialog = new Microsoft.Win32.OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                ModelPathTextBox.Text = openFileDialog.FileName;
                _settingsManager.SaveModelPath(openFileDialog.FileName);
                Manager.RecognizerModelPath = openFileDialog.FileName;
                FIPan.IsEnabled = true;
                FDPan.IsEnabled = true;
                Manager.recognizer.Read(Manager.RecognizerModelPath);
            }
            else
            {
                MessageBox.Show("Не удалось загрузить файл!");
            }
        }


    }
}