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

namespace Face_Detect_System_Test.Pages
{
    /// <summary>
    /// Логика взаимодействия для ModelTrainingPage.xaml
    /// </summary>
    public partial class ModelTrainingPage : Page
    {
        private AesEncryption aesEncryption = new AesEncryption(Manager.key, Manager.iv);
        private ModelTraining MDTrain = new ModelTraining();
        private const string pathYuNetModel = "H:\\face_detection_yunet_2023mar.onnx"; //путь до модели нейронной сети YuNet
        private List<Mat> faces = new List<Mat>();
        private string filePath;
        private PersonInfo currentPerson = new PersonInfo();

        public ModelTrainingPage()
        {
            InitializeComponent();
            var allPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.ToList();
            string decryptPerson;
            foreach (var person in allPerson)
            {
                decryptPerson = aesEncryption.Decrypt(person.LastName) + " " + aesEncryption.Decrypt(person.FirstName) + " " + aesEncryption.Decrypt(person.Patronymic);
                AllPersonComboBox.Items.Add(decryptPerson);
            }

            AddNewPersonBlock.Visibility = Visibility.Hidden;

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
                if (!string.IsNullOrEmpty(myopenFileDialog.FileName))
                {
                    if (myopenFileDialog.FileName.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase))
                    {
                        try
                        {
                            // Проверяем существует ли файл
                            if (File.Exists(myopenFileDialog.FileName))
                            {
                                filePath = myopenFileDialog.FileName;
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
                    currentPerson.LastName = aesEncryption.Encrypt(currentPerson.LastName);
                    currentPerson.FirstName = aesEncryption.Encrypt(currentPerson.FirstName);
                    currentPerson.Patronymic = aesEncryption.Encrypt(currentPerson.Patronymic);
                    allPerson = allPerson.Where(p => p.LastName == currentPerson.LastName & p.FirstName == currentPerson.FirstName & p.Patronymic == currentPerson.Patronymic & p.Birthday == currentPerson.Birthday).ToList();

                    if (allPerson.Count == 0)
                    {
                        if (currentPerson.ID == 0)
                        {
                            PersonInfoForFaceRecEntities.GetContext().PersonInfo.Add(currentPerson);
                        }

                        PersonInfoForFaceRecEntities.GetContext().SaveChanges();
                        MessageBox.Show("Информация добавлена в базу данных!");

                        Update();
                    }
                    else
                    {
                        MessageBox.Show("В базе данных уже существует человек с такими данными!");
                    }
                }
                catch(Exception ex)
                {
                    MessageBox.Show("Ошибка при добавлении!");
                    MessageBox.Show(ex.ToString());
                }
            }
            
        }

        private void Update()
        {
            AllPersonComboBox.Items.Clear();
            var allPerson = PersonInfoForFaceRecEntities.GetContext().PersonInfo.ToList();
            string decryptPerson;
            foreach (var person in allPerson)
            {
                decryptPerson = aesEncryption.Decrypt(person.LastName) + " " + aesEncryption.Decrypt(person.FirstName) + " " + aesEncryption.Decrypt(person.Patronymic);
                AllPersonComboBox.Items.Add(decryptPerson);
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

            if (AllPersonComboBox.SelectedValue == null)
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
            int label = AllPersonComboBox.SelectedIndex;
            await Task.Run(() =>
            {
                _ = Application.Current.Dispatcher.InvokeAsync(() =>
            MainWindow.Instance.trainProcess());
                faces = MDTrain.FacesDetect(filePath, pathYuNetModel);
                MDTrain.ModelTrain(Manager.RecognizerModelPath, faces, label);
            });
            TrainingProcessStackPanel.Visibility = Visibility.Hidden;
            TrainingProcessText.Visibility = Visibility.Hidden;
            _ = Application.Current.Dispatcher.InvokeAsync(() =>
            MainWindow.Instance.Update());
            MessageBox.Show("Модель обучена!");
        }

        private void AddNewPerson_Click(object sender, RoutedEventArgs e)
        {
            AddNewPersonBlock.Visibility = Visibility.Visible;
        }

    }
}
