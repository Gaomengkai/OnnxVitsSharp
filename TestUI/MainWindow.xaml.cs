using System;
using System.Collections.Generic;
using System.ComponentModel;
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

namespace TestUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    /// <summary>
    /// Interaction logic for Window1.xaml
    /// </summary>
    public partial class Window1 : Window
    {
        public Window1()
        {
            InitializeComponent();
            DataContext = new MainWindowViewModel();
        }
    }

    public class PhoneBookEntry
    {
        public string Name { get; set; }
        public PhoneBookEntry(string name)
        {
            Name = name;
        }
    }

    public class ConnectionViewModel : INotifyPropertyChanged
    {

        private string _name;

        public ConnectionViewModel(string name)
        {
            _name = name;
            IList<PhoneBookEntry> list = new List<PhoneBookEntry>
                                             {
                                                 new PhoneBookEntry("test"),
                                                 new PhoneBookEntry("test2")
                                             };
            _phonebookEntries = new CollectionView(list);
        }
        private readonly CollectionView _phonebookEntries;
        private string _phonebookEntry;

        public CollectionView PhonebookEntries
        {
            get { return _phonebookEntries; }
        }

        public string PhonebookEntry
        {
            get { return _phonebookEntry; }
            set
            {
                if (_phonebookEntry == value) return;
                _phonebookEntry = value;
                OnPropertyChanged("PhonebookEntry");
            }
        }

        public string Name
        {
            get { return _name; }
            set
            {
                if (_name == value) return;
                _name = value;
                OnPropertyChanged("Name");
            }
        }
        private void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        public event PropertyChangedEventHandler? PropertyChanged;
    }

    public class MainWindowViewModel
    {
        private readonly CollectionView _connections;

        public MainWindowViewModel()
        {
            IList<ConnectionViewModel> connections = new List<ConnectionViewModel>
                                                          {
                                                              new ConnectionViewModel("First"),
                                                              new ConnectionViewModel("Second"),
                                                              new ConnectionViewModel("Third")
                                                          };
            _connections = new CollectionView(connections);
        }

        public CollectionView Connections
        {
            get { return _connections; }
        }
    }
}
