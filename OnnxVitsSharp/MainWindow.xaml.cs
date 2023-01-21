using HandyControl.Controls;
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

namespace OnnxVitsSharp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private MainWindowViewModel vm;
        public MainWindow()
        {
            InitializeComponent();
            vm = new MainWindowViewModel();
            DataContext = vm;
        }

        private void BtnBrowse_Button_Click(object sender, RoutedEventArgs e)
        {
            this.vm.speakers.Add(new Speaker(123, "123333"));
            this.vm.Speakers.Refresh();
        }
    }
    public class Speaker
    {
        public int _id { get; set; }
        public string _name { get; set; }
        public Speaker(int id, string name) { _id = id; _name = name; }
    }
    public class MainWindowViewModel
    {
        public List<Speaker> speakers = new();
        private CollectionView collectionSpeakers;
        public MainWindowViewModel()
        {
            this.speakers.Add(new Speaker(1,"1111"));
            this.speakers.Add(new Speaker(2,"2222"));
            collectionSpeakers = new CollectionView(this.speakers);

        }
        public CollectionView Speakers
        {
            get => this.collectionSpeakers;
        }
    }

}
