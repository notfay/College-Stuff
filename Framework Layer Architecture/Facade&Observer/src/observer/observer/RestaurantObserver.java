package observer.observer;

import model.Order;

public class RestaurantObserver implements Observer{

	@Override
	public void showNotification(Order order) {
		// TODO Auto-generated method stub
		System.out.println("[Restaurant]: A new order received");
		System.out.println("Ordered by: "+order.getUsername());
		System.out.println("Item name: "+order.getOrderItem());
	}

}
