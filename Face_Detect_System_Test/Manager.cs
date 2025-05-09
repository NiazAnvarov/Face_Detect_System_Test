using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Controls;
using System.IO;
using Emgu.CV.Face;

namespace Face_Detect_System_Test
{
    internal class Manager
    {
        public static Frame MainFrame { get; set; }
        public static string RecognizerModelPath;
        public static LBPHFaceRecognizer recognizer = new LBPHFaceRecognizer();

        private const string SETTINGS_FILE = "face_recognition_settings.json";

        public void SaveModelPath(string path)
        {
            var settings = new FaceRecognitionSettings { ModelPath = path };
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(settings, options);
            File.WriteAllText(SETTINGS_FILE, json);
        }

        public string LoadModelPath()
        {
            if (!File.Exists(SETTINGS_FILE))
                return string.Empty;
            
            var json = File.ReadAllText(SETTINGS_FILE);
            var settings = JsonSerializer.Deserialize<FaceRecognitionSettings>(json);
            return settings?.ModelPath ?? string.Empty;
        }
    }
}
