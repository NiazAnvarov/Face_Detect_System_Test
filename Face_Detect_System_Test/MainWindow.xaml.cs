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
using System.Linq;


namespace Face_Detect_System_Test
{
    public partial class MainWindow : Window
    {

        private FaceIdentifyPage FIPage;
        private FaceDetectPage FDPage;
        private ModelTrainingPage MTPage;
        private readonly Manager _settingsManager;
        private static string filePath = "keys.txt";

        public static MainWindow Instance { get; private set; }

        private FacesDetect facesDetect = new FacesDetect();

        public MainWindow()
        {
            InitializeComponent();
            Instance = this;

            if (File.Exists(filePath))
            {
                string[] lines = File.ReadAllLines(filePath);
                if (lines.Length >= 2)
                {
                    Manager.key = lines[0].Trim();
                    Manager.iv = lines[1].Trim();
                }
                else
                {
                    Console.WriteLine("Недостаточно данных в файле для инициализации ключа и IV.");
                }
            }
            else
            {
                Console.WriteLine($"Файл не найден: {filePath}");
            }

            _settingsManager = new Manager();

            //Загружаем сохрненный путь при запуске
            string savedPath = _settingsManager.LoadModelPath();
            if (!string.IsNullOrEmpty(savedPath))
            {
                try
                {
                    Manager.recognizer.Read(savedPath);
                    Manager.RecognizerModelPath = savedPath;
                    WarningMessageBlock.Visibility = Visibility.Hidden;
                    FIPan.IsEnabled = true;
                    FDPan.IsEnabled = true;
                    FIPan_MouseDown(null, null);
                }
                catch
                {
                    MTPan_MouseDown(null, null);
                    FIPan.IsEnabled = false;
                    FDPan.IsEnabled = false;
                    //System.Windows.MessageBox.Show("Ошибка при загрузке модели!");
                }
            }
            else
            {
                WarningMessageBlock.Visibility = Visibility.Visible;
                MTPan_MouseDown(null, null);
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

            if (FIPage != null)
                return;
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

            if (FDPage != null)
                return;
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

            if (MTPage != null)
                return;
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

            MTPage = new ModelTrainingPage();
            MainFrame.Navigate(MTPage);
            Manager.MainFrame = MainFrame;
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);
            facesDetect?.Dispose();

        }

        private void RecModelLoadBtn_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog openFileDialog = new Microsoft.Win32.OpenFileDialog();

            try
            {
                if (openFileDialog.ShowDialog() == true)
                {

                    if (!ValidateFilePath(openFileDialog.FileName))
                    {
                        System.Windows.MessageBox.Show("Путь не должен содержать русские буквы и пробелы!",
                            "Ошибка", (MessageBoxButton)System.Windows.Forms.MessageBoxButtons.OK, (MessageBoxImage)System.Windows.Forms.MessageBoxIcon.Error);
                        return;
                    }

                    Manager.recognizer.Read(openFileDialog.FileName);
                    _settingsManager.SaveModelPath(openFileDialog.FileName);
                    Manager.RecognizerModelPath = openFileDialog.FileName;
                    FIPan.IsEnabled = true;
                    FDPan.IsEnabled = true;
                }
                else
                {
                    System.Windows.MessageBox.Show("Не удалось загрузить файл!");
                }
            }
            catch
            {
                System.Windows.MessageBox.Show("Ошибка при загрузке модели!");
            }
        }

        // Метод для проверки пути
        private bool ValidateFilePath(string path)
        {
            if (string.IsNullOrEmpty(path))
                return false;

            return !path.Any(c =>
                char.IsLetter(c) && (c > 127 ||
                (c >= 0x0400 && c <= 0x04FF)) ||  // Диапазон кириллицы
                char.IsWhiteSpace(c));
        }

        public void trainProcess()
        {
            FIPan.IsEnabled = false;
            FDPan.IsEnabled = false;
        }

        public void Update()
        {
            string savedPath = _settingsManager.LoadModelPath();
            if (!string.IsNullOrEmpty(savedPath))
            {
                FIPan.IsEnabled = true;
                FDPan.IsEnabled = true;
                try
                {
                    Manager.recognizer.Read(savedPath);
                    Manager.RecognizerModelPath = savedPath;
                    WarningMessageBlock.Visibility = Visibility.Hidden;
                }
                catch
                {
                    MessageBox.Show("Ошибка при загрузке модели!");
                }
            }
            else
            {
                WarningMessageBlock.Visibility = Visibility.Visible;
                FIPan.IsEnabled = false;
                FDPan.IsEnabled = false;
            }
        }
    }
}