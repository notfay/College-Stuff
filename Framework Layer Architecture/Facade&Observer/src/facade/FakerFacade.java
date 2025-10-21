package facade;

import java.util.ArrayList;

import model.Order;
import observer.observable.OrderPublisher;
import observer.observer.DriverObserver;
import observer.observer.RestaurantObserver;

public class FakerFacade {
	
	public OrderPublisher publisher;
	public FakerFacade() {
		this.publisher = new OrderPublisher();
		
		// nambahin observer restaurant
		publisher.addObserver(new RestaurantObserver());
		// nambahin oberserver driver
		publisher.addObserver(new DriverObserver());
	}
	
	//ngambil semua orderan
	public ArrayList<Order> getAllOrders(){
		return publisher.getOrder();
	}
	
	// place order
	public void placeOrder(String name, String OrderItem) {
		Order order = new Order(name, OrderItem);
		publisher.addOrder(order);
	}
	
	
}
