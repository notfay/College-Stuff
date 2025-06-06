package observer.observer;

import model.Order;

public class DriverObserver implements Observer{

	@Override
	public void showNotification(Order order) {
		// TODO Auto-generated method stub
		System.out.println("[Driver]: A new order was recieved from the restaurant");
		System.out.println("Ordered by: "+ order.getUsername());
		System.out.println("Item name: "+order.getOrderItem());
	}

}
