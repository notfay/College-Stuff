package observer.observer;

import model.Order;

public interface Observer {
	
	//satu class order kita mau apply
	public void showNotification(Order order);
}
