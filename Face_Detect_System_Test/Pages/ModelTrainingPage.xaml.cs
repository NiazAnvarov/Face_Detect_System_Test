using Emgu.CV.Structure;
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
using Emgu.CV.Features2D;
using System.Windows.Media.Media3D;
using System.Diagnostics;
using System.Collections.ObjectModel;
using System.IO;

namespace Face_Detect_System_Test.Pages
{
    /// <summary>
    /// Логика взаимодействия для ModelTrainingPage.xaml
    /// </summary>
    public partial class ModelTrainingPage : Page
    {
        public ObservableCollection<PhotoItem> Photos { get; } = new ObservableCollection<PhotoItem>();
        private ModelTraining MDTrain = new ModelTraining();
        private const string pathYuNetModel = "H:\\face_detection_yunet_2023mar.onnx"; //путь до модели нейронной сети YuNet
        private List<Mat> faces = new List<Mat>();
        private string filePath;

        public ModelTrainingPage()
        {
            InitializeComponent();

            
            
            
            
            //Console.WriteLine(faces.Count());
            //foreach (Mat face in faces)
            //{
            //    AddPhoto(BitmapSourceConvert(face));
            //    Console.WriteLine("1");
            //}

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

        private void AddPhoto(BitmapSource bitmap)
        {
            Photos.Add(new PhotoItem(bitmap));
        }

        private void VideoLoadBtn_Click(object sender, RoutedEventArgs e)
        {
            var myopenFileDialog = new Microsoft.Win32.OpenFileDialog();
            if (myopenFileDialog.ShowDialog() == true)
            {
                filePath = myopenFileDialog.FileName;

                if (!string.IsNullOrEmpty(filePath))
                {
                    if (filePath.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase))
                    {
                        try
                        {
                            // Проверяем существует ли файл
                            if (File.Exists(filePath))
                            {
                                // Здесь можно использовать filePath для дальнейшей работы с файлом
                                System.Windows.MessageBox.Show($"Выбранный файл: {filePath}");
                            }
                            else
                            {
                                System.Windows.MessageBox.Show("Файл не найден!");
                            }
                        }
                        catch (Exception ex)
                        {
                            System.Windows.MessageBox.Show($"Ошибка при работе с файлом: {ex.Message}");
                        }
                    }
                    else
                    {
                        System.Windows.MessageBox.Show("Не верный формат файла! Требуется формат MP4.");
                    }
                }
            }
        }

        private void PersonAddBtn_Click(object sender, RoutedEventArgs e)
        {
            faces = MDTrain.FacesDetect(filePath, pathYuNetModel);
            MDTrain.ModelTrain("H:\\testRecMod.xml", faces, 0);
        }
    }
}
