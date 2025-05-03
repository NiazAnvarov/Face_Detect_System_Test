using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace Face_Detect_System_Test
{
    public class PhotoItem
    {
        public BitmapSource Image { get; set; }
        public PhotoItem(BitmapSource image)
        {
            Image = image;
        }
    }


}
