package observer.observable;

import java.util.ArrayList;

import model.Order;
import observer.observer.Observer;

public class OrderPublisher {
	//task
	//1. Mengatur Orderan (view order, add order)
	//2. Mengatur notification (notify all, add observer)
	
	// Logika broadcast ke semua
	//1. Masukin dulu / daftarin observer ke koleksi yang bakal di broadcast
	//2. Kirim ke semua yang udah masuk ke list broadcast
	
	public ArrayList<Order> orders;
	public ArrayList<Observer> observers;
	
	public OrderPublisher() {
		this.orders = new ArrayList<>();
		this.observers = new ArrayList<>();
	}

	public void addOrder(Order order) {
		System.out.println("New Order Created");
		orders.add(order);
		notifyAll(order);
	}
	
	public ArrayList<Order> getOrder(){
		return orders;
	}
	
	public void addObserver(Observer observer) {
		observers.add(observer);
	}
	
	public void notifyAll(Order order) {
		for(Observer observer : observers) {
			observer.showNotification(order);
		}
	}
}
