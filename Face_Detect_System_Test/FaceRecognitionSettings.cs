using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json.Serialization;

namespace Face_Detect_System_Test
{
    internal class FaceRecognitionSettings
    {
        [JsonPropertyName("model_path")]
        public string ModelPath {  get; set; }
    }
}
