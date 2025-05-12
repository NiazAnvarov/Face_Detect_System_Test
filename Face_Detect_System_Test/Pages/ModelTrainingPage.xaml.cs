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
using Microsoft.Win32;
using System.Windows.Threading;
//using System.Windows.Forms;

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
        private PersonInfo currentPerson = new PersonInfo();
        private string selectedPath;

        public ModelTrainingPage()
        {
            InitializeComponent();
            var allPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.Select(p =>  p.LastName + " " + p.FirstName + " " + p.Patronymic).ToList();
            foreach (var person in allPerson)
            {
                AllPersonComboBox.Items.Add(person);
            }
            //faces = MDTrain.FacesDetect("C:\\Users\\niaza\\Pictures\\Camera Roll\\WIN_20250503_16_09_43_Pro.mp4", pathYuNetModel);
            //MDTrain.ModelTrain("H:\\testModelForRec.xml", faces, 0);
            if (string.IsNullOrEmpty(Manager.RecognizerModelPath))
            {
                ModelDirectoryPathTextBlock.Visibility = Visibility.Visible;
                ModelDirectoryBtn.Visibility = Visibility.Visible;
            }
            else
            {
                ModelDirectoryPathTextBlock.Visibility = Visibility.Hidden;
                ModelDirectoryBtn.Visibility = Visibility.Hidden;
            }
            BirthdayDP.SelectedDate = new DateTime(2024, 1, 1);
            PersonLastName.PreviewTextInput += new TextCompositionEventHandler(PersonLastName_PreviewTextInput);
            PersonFirstName.PreviewTextInput += new TextCompositionEventHandler(PersonFirstName_PreviewTextInput);
            PersonPatronymic.PreviewTextInput += new TextCompositionEventHandler(PersonPatronymic_PreviewTextInput);
            DataContext = currentPerson;
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

            StringBuilder error = new StringBuilder();
            if (string.IsNullOrWhiteSpace(currentPerson.LastName))
            {
                error.AppendLine("Введите Фамилию!");
            }
            if (string.IsNullOrWhiteSpace(currentPerson.FirstName))
            {
                error.AppendLine("Введите Имя!");
            }
            if (string.IsNullOrWhiteSpace(currentPerson.Activity))
            {
                error.AppendLine("Введите деятельность!");
            }
            if(filePath ==  null)
            {
                error.AppendLine("Загрузите видеофайл!");
            }
            if(error.Length > 0)
            {
                MessageBox.Show(error.ToString());
                return;
            }
            else
            {
                try
                {
                    currentPerson.Birthday = (DateTime)BirthdayDP.SelectedDate;
                    int Label = PersonInfoForFaceRecEntities.GetContext().PersonInfo.OrderByDescending(p => p.ID).FirstOrDefault().ID + 1;
                    var allPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.ToList();
                    allPerson = allPerson.Where(p => p.LastName == currentPerson.LastName & p.FirstName == currentPerson.FirstName & p.Patronymic == currentPerson.Patronymic).ToList();

                    if (allPerson.Count == 0)
                    {
                        if (currentPerson.ID == 0)
                        {
                            PersonInfoForFaceRecEntities.GetContext().PersonInfo.Add(currentPerson);
                        }

                        PersonInfoForFaceRecEntities.GetContext().SaveChanges();
                        MessageBox.Show("Информация добавлена в базу данных!");
                        TrainingProcessStackPanel.Visibility = Visibility.Visible;
                        TrainingProcessText.Visibility = Visibility.Visible;
                        faces = MDTrain.FacesDetect(filePath, pathYuNetModel);
                        MDTrain.ModelTrain(Manager.RecognizerModelPath, faces, Label);
                        TrainingProcessStackPanel.Visibility = Visibility.Hidden;
                        TrainingProcessText.Visibility = Visibility.Hidden;
                        MessageBox.Show("Модель обучена!");
                    }
                    else
                    {
                        MessageBox.Show("В базе данных уже существует информация о таком человеке!");
                    }
                }
                catch(Exception ex)
                {
                    MessageBox.Show("Ошибка при добавлении!");
                    MessageBox.Show(ex.ToString());
                }
            }
            
        }

        private void PersonLastName_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            if (!Char.IsLetter(e.Text[0]))
            {
                e.Handled = true;
            }
        }

        private void PersonFirstName_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            if (!Char.IsLetter(e.Text[0]))
            {
                e.Handled = true;
            }
        }

        private void PersonPatronymic_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            if (!Char.IsLetter(e.Text[0]))
            {
                e.Handled = true;
            }
        }

        private async void ModelTrainBtn_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(Manager.RecognizerModelPath))
            {
                if (selectedPath != null)
                {
                    Manager.RecognizerModelPath = selectedPath + "recModel.xml";
                }
                else
                {
                    MessageBox.Show("Выберите папку куда будет сохранена модель!");
                    return;
                }
            }
            if(AllPersonComboBox.SelectedValue == null)
            {
                MessageBox.Show("Выберите человека из списка!");
                return;
            }
            if (filePath == null)
            {
                MessageBox.Show("Загрузите видеофайл!");
                return;
            }
            
            TrainingProcessStackPanel.Visibility = Visibility.Visible;
            TrainingProcessText.Visibility = Visibility.Visible;
            await Dispatcher.InvokeAsync(() => { }, DispatcherPriority.Render);
            faces = MDTrain.FacesDetect(filePath, pathYuNetModel);
            MDTrain.ModelTrain(Manager.RecognizerModelPath, faces, AllPersonComboBox.SelectedIndex);
            TrainingProcessStackPanel.Visibility = Visibility.Hidden;
            TrainingProcessText.Visibility = Visibility.Hidden;
            MessageBox.Show("Модель обучена!");
        }

        private void ModelDirectoryBtn_Click(object sender, RoutedEventArgs e)
        {

            System.Windows.Forms.FolderBrowserDialog myFolderDialog = new System.Windows.Forms.FolderBrowserDialog();

            // Настройка диалога
            myFolderDialog.Description = "Выберите папку";
            

            // Показ диалога
            if (myFolderDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                if (myFolderDialog.SelectedPath != null)
                {
                    // Проверяем наличие русских букв или пробелов
                    bool containsRussianOrSpace = myFolderDialog.SelectedPath.Any(c =>
                        char.IsLetter(c) && (c > 127 ||
                        (c >= 0x0400 && c <= 0x04FF)) ||  // Диапазон кириллицы
                        char.IsWhiteSpace(c));

                    if (containsRussianOrSpace)
                    {
                        MessageBox.Show("Путь не должен содержать русские буквы и пробелы!",
                            "Ошибка", (MessageBoxButton)System.Windows.Forms.MessageBoxButtons.OK, (MessageBoxImage)System.Windows.Forms.MessageBoxIcon.Error);
                        return;
                    }
                    selectedPath = myFolderDialog.SelectedPath;
                    ModelDirectoryPathTextBlock.Text = selectedPath;

                }
                else
                {
                    MessageBox.Show("Произошла ошибка при выборе папки!");
                }
            }
        }
    }
}
